from torch.utils.data import Dataset
import numpy as np
from functools import lru_cache
from torchvision import transforms
import SimpleITK as sitk
from pathlib import Path
import glob
import os
from os import path
import torch
import torch.utils.data
import torch.nn as nn

import tqdm
from scipy import ndimage
import csv
import cv2
from scipy.ndimage.interpolation import map_coordinates, rotate
from scipy.ndimage.filters import gaussian_filter
import math
import pandas
import xlsxwriter as writer
import scipy.misc
from sklearn.metrics import cohen_kappa_score
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter, LinearSegmentedColormap, ListedColormap

import sys


def basename(arg):
    try:
        return os.path.splitext(os.path.basename(arg))[0]
    except Exception as e:
        if isinstance(arg, list):
            return [basename(el) for el in arg]
        else:
            raise e

def loadPNG(fname):
    img = cv2.imread(fname)  # [z,y,x]
    im = np.asarray(img[:, :, 0]) #you only need to give back one color channel
    return im

def loadMHD(fname):
    img = sitk.ReadImage(str(fname)) #[z,y,x]
    spacing = img.GetSpacing()[::-1]
    offset = img.GetOrigin()[::-1]
    img = sitk.GetArrayFromImage(img)
    spacing = np.asarray(spacing)
    offset = np.asarray(offset)
    return img, spacing, offset

def saveImage(fname, arr, spacing=None, dtype=np.float32):
    if type(spacing) == type(None):
        spacing = np.ones((len(arr.shape),))
    img = sitk.GetImageFromArray(arr.astype(dtype))
    img.SetSpacing(spacing[::-1])
    sitk.WriteImage(img, fname, True)

def loadCephalometric(data_dir):
    im_folder = data_dir + "/images_png/"
    lab_dir = r"/home/julia/landmark_detection/paper/data/labels/Cephalometric/origineelAverage/"
    image_fnames = glob.glob(path.join(im_folder, '*.png'))
    images = list()
    landmarks = list()
    bns = list()
    for f in tqdm.tqdm(image_fnames):
        im = loadPNG(f)
        bn = basename(f)
        lms = np.loadtxt(lab_dir + bn + ".txt") #[:-4]
        lms = np.asarray(lms)
        temp = np.copy(lms[:, 0]) #switch x and y
        lms[:, 0] = lms[:, 1]
        lms[:, 1] = temp
        images.append(im)
        landmarks.append(lms)
        bns.append(bn)
    images = np.asarray(images)
    landmarks = np.asarray(landmarks)
    return images, landmarks, bns

def loadCCTA(data_dir, vs=0.1):
    if vs ==0.5:
        im_folder = data_dir + "/images_VS05/"
        lab_dir = r"/home/julia/landmark_detection/paper/data/labels/CCTA/VS_05/"
    elif vs == 1.5:
        im_folder = data_dir + "/images_VS15/"
        lab_dir = r"/home/julia/landmark_detection/paper/data/labels/CCTA/VS_15/"
    image_fnames = glob.glob(path.join(im_folder, '*.nii.gz'))
    image_fnames.sort()

    images = list()
    landmarks = list()
    bns = []
    for f in tqdm.tqdm(image_fnames):
        im, spacing, offset = loadMHD(f)
        bn = basename(f)
        bns.append(bn)
        lms = np.loadtxt(lab_dir + bn[:-4] + ".txt") #LH, NCH, RH, NCRC, NCLC, LRC, RO, LO
        lms = np.asarray(lms)
        images.append(im)
        landmarks.append(lms)
    return np.asarray(images), np.asarray(landmarks), bns

def loadBulbusMRI(data_dir, chunk_size):
    # lm == -1 --> both lm
    # lm == 0 --> links
    # lm == 1 --> rechts

    im_dir = data_dir + "/images_VS047/"
    lab_dir = r"/home/julia/landmark_detection/paper/data/labels/Bulbus/isotropic_lm/"
    image_fnames = glob.glob(path.join(im_dir, '*.nii.gz'))

    images = list()
    landmarks = list()
    bns = []

    for f in tqdm.tqdm(image_fnames):
        im, spacing, offset = loadMHD(f)
        z_pad = chunk_size-im.shape[0]
        if z_pad > 0:
            im = np.pad(im, ((0, z_pad+1), (0, 0), (0, 0)), mode="constant")
        bn = basename(f)
        bns.append(bn)
        lm_left = np.loadtxt(lab_dir + bn[:-4] + "_Links.txt")
        lm_right = np.loadtxt(lab_dir + bn[:-4] + "_Rechts.txt")
        lms = (lm_left, lm_right)
        landmarks.append(lms)
        images.append(im)
    return np.asarray(images), np.asarray(landmarks), bns


def create_landmark_classmask(shape, spacing, landmarks):
    class_mask = np.zeros([len(landmarks)] + list(shape), bool)
    idcs = (landmarks / spacing).astype(int)
    valid_landmarks = np.logical_and((idcs >= 0).all(1), np.less(idcs, shape).all(1))
    idcs_raveled = np.ravel_multi_index(idcs.T, dims=class_mask.shape[1:], mode='clip')
    idcs_raveled = idcs_raveled + np.cumsum([0]+[np.product(class_mask.shape[1:])]*(len(landmarks)-1))
    idcs_raveled = idcs_raveled[valid_landmarks]
    class_mask.flat[idcs_raveled] = True
    return class_mask

def create_landmark_grid2D(shape, spacing, landmarks):
    extent = spacing * shape
    linear_coordinates = [np.linspace(0, ext, shp, endpoint=False) for ext, shp in zip(extent, shape)]
    grid = np.stack(np.meshgrid(*linear_coordinates, indexing='ij'))
    center_offset = spacing/2
    grid = grid + center_offset[:, None, None]
    grid = landmarks[:, :, None, None] - grid[None]
    return grid

def create_landmark_grid3D(shape, spacing, landmarks):
    extent = spacing * shape
    linear_coordinates = [np.linspace(0, ext, shp, endpoint=False) for ext, shp in zip(extent, shape)]
    grid = np.stack(np.meshgrid(*linear_coordinates, indexing='ij'))
    center_offset = spacing/2
    grid = grid + center_offset[:, None, None, None]
    grid = landmarks[:, :, None, None, None] - grid[None]
    return grid

def log(x, logfn=np.log10):
    epsilon = 1
    mask = x < 0
    y = np.empty_like(x, np.float32)
    y[~mask] = logfn(x[~mask]+epsilon)
    y[mask] = -logfn(-1*(x[mask]-epsilon))
    return y

def BatchIterator_2D(images, landmarks, batch_size, chunk_size, nclass, rf_list, landmark_grids, regr_trans):
    assert (len(chunk_size) == 2)
    assert (len(images) == len(landmarks))
    assert (np.remainder(chunk_size, (rf_list[-1,1], rf_list[-1,1])).any() == False)  # chunksize moet een veelvoud van de patch_size zijn.
    rs_data = np.random.RandomState(123)

    while True:
        sample_indices = rs_data.randint(len(images), size=batch_size)
        batch = np.empty((batch_size, 1) + chunk_size, dtype=np.float32)
        targets_dist = []
        targets_class= []
        for r in range(len(rf_list)):
            targets_dist.append(np.empty((batch_size, nclass, 2) + tuple(landmark_grids[r]), dtype=np.float32))
            targets_class.append(np.empty((batch_size, nclass) + tuple(landmark_grids[r]), dtype=np.float32))

        for idx in range(len(sample_indices)):
            n = sample_indices[idx]
            image=images[n]

            #if chunk size is bigger than image, padding is necessary
            path_y = max(0, chunk_size[0]-image.shape[0]+1)
            path_x = max(0, chunk_size[1]-image.shape[1]+1)
            image = np.pad(image, ((0, path_y), (0, path_x)), mode="constant")

            y_space, x_space = np.subtract(image.shape, chunk_size)
            y_offset = rs_data.randint(y_space)
            x_offset = rs_data.randint(x_space)
            chunk = image[y_offset:y_offset + chunk_size[0],
                    x_offset:x_offset + chunk_size[1]]

            landmarks_new =landmarks[n]-(y_offset, x_offset)
            for r in range(len(rf_list)):
                ps = np.array((rf_list[r,1], rf_list[r,1]))
                chunk_shape = chunk_size // ps
                landmark_grid = create_landmark_grid2D(chunk_shape, ps, landmarks_new)
                class_mask = create_landmark_classmask(chunk_shape, ps, landmarks_new)
                dist=log(landmark_grid, regr_trans)
                targets_dist[r][idx] = dist
                targets_class[r][idx] = class_mask
            batch[idx] = chunk
        yield batch, targets_dist, targets_class

def BatchIterator_3D(images, landmarks, batch_size, chunk_size, nclass, rf_list, landmark_grids, regr_trans):
    assert (len(chunk_size) == 3)
    assert (len(images) == len(landmarks))
    assert (np.remainder(chunk_size, (rf_list[-1,1], rf_list[-1,1], rf_list[-1,1])).any() == False)  # chunksize moet een veelvoud van de patch_size zijn.
    rs_data = np.random.RandomState(123)

    while True:
        sample_indices = rs_data.randint(len(images), size=batch_size)
        batch = np.empty((batch_size, 1) + chunk_size, dtype=np.float32)
        targets_dist = []
        targets_class= []
        for r in range(len(rf_list)):
            targets_dist.append(np.empty((batch_size, nclass, 3) + tuple(landmark_grids[r]), dtype=np.float32))
            targets_class.append(np.empty((batch_size, nclass) + tuple(landmark_grids[r]), dtype=np.float32))

        for idx in range(len(sample_indices)):
            n = sample_indices[idx]
            image=images[n]

            #if chunk size is bigger than image, padding is necessary
            path_z = max(0, chunk_size[0]-image.shape[0]+1)
            path_y = max(0, chunk_size[1]-image.shape[1]+1)
            path_x = max(0, chunk_size[2]-image.shape[2]+1)
            image = np.pad(image, ((0, path_z), (0, path_y), (0, path_x)), mode="constant")

            z_space, y_space, x_space = np.subtract(image.shape, chunk_size)
            z_offset = rs_data.randint(z_space)
            y_offset = rs_data.randint(y_space)
            x_offset = rs_data.randint(x_space)
            chunk = image[z_offset:z_offset + chunk_size[0],y_offset:y_offset + chunk_size[1],
                    x_offset:x_offset + chunk_size[2]]

            landmarks_new =landmarks[n]-(z_offset, y_offset, x_offset)
            for r in range(len(rf_list)):
                ps = np.array((rf_list[r,1], rf_list[r,1], rf_list[r,1]))
                chunk_shape = chunk_size // ps
                landmark_grid = create_landmark_grid3D(chunk_shape, ps, landmarks_new)
                class_mask = create_landmark_classmask(chunk_shape, ps, landmarks_new)
                dist=log(landmark_grid, regr_trans)
                targets_dist[r][idx] = dist
                targets_class[r][idx] = class_mask
            batch[idx] = chunk
        yield batch, targets_dist, targets_class
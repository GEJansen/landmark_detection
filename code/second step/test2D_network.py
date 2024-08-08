from __future__ import print_function
from networks_2D import MultiFineTune
import torch
from torchvision import transforms
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np
import math
import xlsxwriter as writer
from os import path
import glob
import time
import pandas
import data_augmentation_2D as data
import sys
epsilon =1
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm
from matplotlib.colors import colorConverter, LinearSegmentedColormap, ListedColormap
import matplotlib

def loadExperimentSettings(fname):
    with open(fname, 'r') as fp:
        args = argparse.Namespace(**yaml.load(fp))
    return args

def create_boxplot(dist_errors, n):
    dist_errors=np.asarray(dist_errors)
    fig, axes = plt.subplots(figsize=(12, 16))
    bplot= axes.boxplot(dist_errors, patch_artist=True)

    # Fill with colors
    cm = plt.cm.get_cmap('rainbow')
    colors = [cm(val / len(dist_errors)) for val in range(len(dist_errors))]

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    axes.set_ylim((0, 10))
    plt.savefig(
        n + ".png",bbox_inches='tight',
                pad_inches=0, transparent=True)
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation network')
    parser.add_argument('--experiment_directory', type=str, help='directory for experiment outputs')
    parser.add_argument('--choosen_iter', type=int, default=0)
    parser.add_argument('--heatmap', action='store_true')
    return parser.parse_args()

def computeCenterpoints(imshape, spacings):
    centerpoints = []
    for s in range(len(spacings)):
        spacing = np.array((spacings[s]))
        shape = imshape // spacing
        extent = spacing * shape
        linear_coordinates = [np.linspace(spacing / 2, ext + spacing / 2, shp, endpoint=False) for ext, shp in
                              zip(extent, shape)]
        centerpoints.append(np.stack(np.meshgrid(*linear_coordinates, indexing='ij')))
    return centerpoints

def computeCenterpointSingle(imshape, spacing):
    spacing = np.array(spacing)
    shape = imshape // spacing
    extent = spacing * shape
    linear_coordinates = [np.linspace(spacing / 2, ext + spacing / 2, shp, endpoint=False) for ext, shp in
                          zip(extent, shape)]
    return np.stack(np.meshgrid(*linear_coordinates, indexing='ij'))


def loginv(x, gr=10):
    mask = x < 0
    y = np.empty_like(x, np.float32)
    y[~mask] = np.power(gr, x[~mask])-epsilon
    y[mask] = -np.power(gr,-1*(x[mask]))+epsilon
    return y

def test_network(network, dataloader, nr_ims, n_class, vs):
    times = []
    distance_errors = np.empty((nr_ims, n_class))
    bns = []

    for imidx, test_batch in tqdm(enumerate(dataloader), total=nr_ims):
        ims = test_batch['image'].type(torch.FloatTensor).cuda()
        lms = test_batch['displacement'].cuda()
        bns.append(test_batch['name'])
        output = network(ims)
        output = output

        for c in range(n_class):
            lm = lms[0, c].cpu().detach().numpy()
            pred = output[0, c].cpu().detach().numpy()
            z_error = (pred[0] - lm[0]) * vs[0]
            y_error = (pred[1] - lm[1]) * vs[1]
            distance_errors[imidx, c] = math.sqrt(((z_error ** 2) + (y_error ** 2)))
            print("Predicted lm: ", pred, " Reference lm: ", lm)
            print("Error: ", distance_errors[imidx, c])
    return times, distance_errors, bns

def test_networkRandC(network, dataloader, nr_ims, n_class, vs, logloss):
    times = []
    distance_errors = np.empty((nr_ims, n_class))
    bns = []

    for imidx, test_batch in tqdm(enumerate(dataloader), total=nr_ims):
        ims = test_batch['image'].type(torch.FloatTensor).cuda()
        lms = test_batch['landmarks'].cuda()
        bns.append(test_batch['name'])
        start = time.time()
        output_cls, output_rgr = network(ims)
        centerpoints = computeCenterpointSingle(ims[0,0].shape, [2])

        for c in range(n_class):
            lm = lms[0, c].cpu().detach().numpy()
            pred_cl = output_cls[0, c].cpu().detach().numpy()
            pred_regr = output_rgr[0, c].cpu().detach().numpy()
            if logloss:
                pred_regr = loginv(pred_regr)
            locs = pred_regr + centerpoints
            predz = np.average(locs[0].flatten(), weights=pred_cl.flatten())
            predy = np.average(locs[1].flatten(), weights=pred_cl.flatten())
            pred = [predz, predy]
            z_error = (predz - lm[0]) * vs[0]
            y_error = (predy - lm[1]) * vs[1]
            distance_errors[imidx, c] = math.sqrt(((z_error ** 2) + (y_error ** 2)))
            # print("Predicted lm: ", pred, " Reference lm: ", lm)
            # print("Error: ", distance_errors[imidx, c])
        times.append(time.time() - start)
    return times, distance_errors, bns

def main(task, dataset, choosen_iter, heatmap):
    exp_dir = args_inp.experiment_directory
    args = loadExperimentSettings(exp_dir + '/settings.yaml')
    exp_name = exp_dir.split("/")[-1]
    patch_size = 16
    RandC = 'RandC' in args.network

    n_class = 19

    if args.lm >= 0:
        n_class=1
        single_lm=True
    try:
        n_resnetblocks = args.n_resnetblocks
    except:
        n_resnetblocks=3


    try:
        log_loss=args.log_loss
    except:
        log_loss=False

    #load data
    network = MultiFineTune(n_class, network_arch=args.network)
    transforms_validation = transforms.Compose(
        [data.TestSubImagesAroundLandmarks(size=patch_size)])

    if 'Cephalometric' in args.data_dir:
        validation_set = data.CephalometricDataset(dataset, transform=transforms_validation, single=False, lm_idx=args.lm)
        vs = (0.1, 0.1)

    # if single_lm:
    #     validation_landmarks = validation_landmarks[:, args.lm]
    #     validation_landmarks = validation_landmarks[:, np.newaxis, :]

    sampler_validation = SequentialSampler(validation_set)

    data_loader_validation = DataLoader(validation_set,
                                      batch_size=1,
                                      sampler=sampler_validation,
                                      num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)
    network.eval()

    if task == "validation":
        choosen_iter="ALL"
        average_errors = np.empty((30, n_class))
        median_errors = np.empty((30, n_class))
        #test network for x iterations on validationset
        for iter in range(20000, 310000, 10000):
            print("Iteration: ", str(iter))
            w = exp_dir + '/model_'+ str(iter) + '.pt'
            if iter == 310000:
                w = exp_dir +'/model_'+ str(iter) + '_final.pt'
            network.load_state_dict(torch.load(w)['model'])
            torch.no_grad()
            if RandC:
                times, distance_errors, bns = test_networkRandC(network, data_loader_validation, len(validation_set), n_class, vs, log_loss)
            else:
                times, distance_errors, bns = test_network(network, data_loader_validation, len(validation_set), n_class, vs)
            for i in range(n_class):
                average_errors[(iter // 10000 - 2), i] = np.average(distance_errors[:, i])
                median_errors[(iter // 10000 - 2), i] = np.median(distance_errors[:, i])
            create_boxplot(distance_errors, save_dir + "_boxplot_CONSEC"+ task + "_" + dataset + "_" + str(iter))
        df = pandas.DataFrame(average_errors)
        df.to_excel(save_dir + "_averageErrors_" + task + "_" + dataset + "CONSEC.xlsx", index=False)
        df = pandas.DataFrame(median_errors)
        df.to_excel(save_dir + "_medianErrors_" + task + "_" + dataset + "CONSEC.xlsx", index=False)


    elif "testing" in task:
        w = exp_dir + '/model_' + str(choosen_iter) + '.pt'
        if choosen_iter == 310000:
            w = exp_dir + '/model_' + str(choosen_iter) + '_final.pt'
        network.load_state_dict(torch.load(w)['model'])
        torch.no_grad()
        if RandC:
            times, distance_errors, bns = test_networkRandC(network, data_loader_validation, len(validation_set), n_class, vs, log_loss)
        else:
            times, distance_errors, bns = test_network(network, data_loader_validation, len(validation_set), n_class, vs)
        df = pandas.DataFrame(distance_errors)
        df.to_excel(save_dir + "_DistanceErrors_" + task + "_" + dataset + "_" + str(choosen_iter) + "CONSEC.xlsx", index=False)
        # df = pandas.DataFrame(times)
        # df.to_excel(save_dir + "_TIMES_" + task + "_" + dataset + "_" + str(choosen_iter) + "CONSEC.xlsx",
                    index=False)
        create_boxplot(distance_errors, save_dir + "_boxplot_CONSEC"+ task + "_" + dataset + "_" + str(choosen_iter))

    df = pandas.DataFrame(bns)
    df.to_excel(save_dir + "_BNS_" + task + ".xlsx", index=False)


if __name__ == '__main__':
    rs = np.random.RandomState(123)
    args_inp = parse_args()
    save_dir = r"/home/julia/landmark_detection/rebuttal_final/results_finetuning/" + args_inp.experiment_directory.split("/")[-1]
    # main(task="validation", dataset = "val", choosen_iter=args_inp.choosen_iter, heatmap=args_inp.heatmap)
    main(task="testing", dataset = "test1", choosen_iter=args_inp.choosen_iter, heatmap=args_inp.heatmap)
    # main(task="testing", dataset = "test2", choosen_iter=args_inp.choosen_iter, heatmap=args_inp.heatmap)
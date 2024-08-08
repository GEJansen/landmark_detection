from __future__ import print_function
from networks_3D import Network
import torch
import numpy as np
import math
import xlsxwriter as writer
from os import path
import glob
import time
import pandas
import data
import sys
epsilon =1
import argparse
import yaml
from pathlib import Path
from figures import create_boxplot, makeheatmaps_mulitlm, makeheatmaps
from create_vector_image import createResultImages3D_onlyax

def loadExperimentSettings(fname):
    with open(fname, 'r') as fp:
        args = argparse.Namespace(**yaml.load(fp))
    return args

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

def loginv(x, gr=10):
    mask = x < 0
    y = np.empty_like(x, np.float32)
    y[~mask] = np.power(gr, x[~mask])-epsilon
    y[mask] = -np.power(gr,-1*(x[mask]))+epsilon
    return y

def test_network(network, rf_list, images, landmarks, heatmap, gt, vs, n_class, bns):
    times = []
    distance_errors = np.empty((len(images), n_class))
    for imidx in range(len(images)):
        print("Image index: ", str(imidx))
        im = images[imidx]
        lms = landmarks[imidx]
        filename = bns[imidx]
        z_padding = rf_list[-1, 1] - (im.shape[0]%rf_list[-1, 1])
        y_padding = rf_list[-1, 1] - (im.shape[1]%rf_list[-1, 1])
        x_padding = rf_list[-1, 1] - (im.shape[2]%rf_list[-1, 1])

        if z_padding == rf_list[-1, 1]:
            z_padding = 0
        if y_padding == rf_list[-1,1]:
            y_padding = 0
        if x_padding == rf_list[-1,1]:
            x_padding = 0
        im = np.pad(im, ((0, z_padding), (0, y_padding), (0, x_padding)), mode="constant")
        centerpoints = computeCenterpoints(im.shape, rf_list[:, 1])

        start = time.time()
        output = network(torch.from_numpy(im[None, None]).type(torch.FloatTensor).cuda())

        averaged_z_overresults = np.zeros(n_class)
        averaged_y_overresults = np.zeros(n_class)
        averaged_x_overresults = np.zeros(n_class)
        for out_idx in range(len(output)):
            pred_class = output[out_idx][0][0].cpu().detach().numpy()
            log_pred_dist = output[out_idx][1][0].cpu().detach().numpy()
            real_pred_dist = loginv(log_pred_dist, gt)

            ##use only groundtruthpositive patches for zooming in
            # pred_class = data.create_landmark_classmask(output[out_idx][0][0].cpu().detach().numpy().shape[-2:], rf_list[out_idx, 1], lms)

            for c in range(n_class):
                locs = real_pred_dist[c] + centerpoints[out_idx]
                pred_z_loc = np.average(locs[0].flatten(), weights=pred_class[c].flatten())
                pred_y_loc = np.average(locs[1].flatten(), weights=pred_class[c].flatten())
                pred_x_loc = np.average(locs[2].flatten(), weights=pred_class[c].flatten())

                averaged_z_overresults[c] += pred_z_loc
                averaged_y_overresults[c] += pred_y_loc
                averaged_x_overresults[c] += pred_x_loc

                # if heatmap: #make and save heatmaps
                #     makeheatmaps(im, lms[c], real_pred_dist[c] + centerpoints[out_idx],
                #              pred_class[c], str(imidx)+"_lm" + str(c), save_dir)
            if heatmap:  # make and save heatmaps
                makeheatmaps_mulitlm(im, lms, real_pred_dist + centerpoints[out_idx],
                             pred_class, str(imidx) + "_lm" + "result_" + str(out_idx), save_dir)

        predlms=[]
        for c_idx in range(n_class):
            pred_z = (averaged_z_overresults[c_idx]/len(output))
            pred_y = (averaged_y_overresults[c_idx]/len(output))
            pred_x = (averaged_x_overresults[c_idx]/len(output))
            lm = lms[c_idx]

            z_error = (pred_z-lm[0])*vs[0]
            y_error = (pred_y-lm[1])*vs[1]
            x_error = (pred_x-lm[2])*vs[2]

            distance_errors[imidx, c_idx]= math.sqrt(((z_error**2)+(y_error**2)+(x_error**2)))
            # print("Predicted lm: ", pred_z, pred_y, pred_x, " Reference lm: ", lm)
            # print("Error: ", distance_errors[imidx, c_idx])
            predlms.append((pred_z, pred_y, pred_x))
            if c_idx == 6:
                if filename == "1.2.840.113704.1.111.3984.1332166750.34.nii" or filename == "1.2.840.113704.1.111.2748.1324365544.22.nii" or filename == "1.2.840.113704.1.111.4120.1321600725.22.nii":
                    createResultImages3D_onlyax(im, lm, (pred_z, pred_y, pred_x), pred_class[6], 8, real_pred_dist[6],
                                                filename,
                                                "/home/julia/landmark_detection/rebuttal_final/images/BAD_RO_")
                if filename == "1.2.840.113704.1.111.2384.1363253958.22.nii" or filename == "1.2.840.113704.1.111.3504.1329474070.34.nii" or filename == "1.2.840.113704.1.111.21472.1349081672.31.nii":
                    createResultImages3D_onlyax(im, lm, (pred_z, pred_y, pred_x), pred_class[6], 8, real_pred_dist[6],
                                                filename,
                                                "/home/julia/landmark_detection/rebuttal_final/images/GOOD_RO_")
        # np.savetxt(r'/home/julia/landmark_detection/paper/data/labels/CCTA/VS_15_EstimatedSL7/'+filename + '.txt', predlms)
        # np.savetxt(r'/home/julia/landmark_detection/paper/data/labels/Bulbus/isotropic_lm_EstimatedSL0/'+filename + '.txt', predlms)
        # np.savetxt(r'/home/julia/landmark_detection/paper/data/labels/CCTA/VS_15_EstimatedML/'+filename + '.txt', predlms)
        times.append(time.time() - start)

    return times, distance_errors



def main(task, dataset, choosen_iter, heatmap):
    exp_dir = args_inp.experiment_directory
    args = loadExperimentSettings(exp_dir + '/settings.yaml')
    exp_name = exp_dir.split("/")[-1]

    if args.regr_trans == 'log':
        gt = 10
    elif args.regr_trans == 'ln':
        gt = np.e

    if 'CCTA' in args.data_dir:
        n_class = 8
    elif 'Bulbus' in args.data_dir:
        n_class = 2
    single_lm = args.single_lm
    # single_lm = False
    if single_lm:
        n_class = 1

    try:
        n_resnetblocks = args.n_resnetblocks
    except:
        n_resnetblocks=3

    #load data
    network, rf_list = Network(args.network, n_class, args.n_convpairs, args.n_downsampling, args.interlayers,
                               args.prod_class, args.sevxsev, args.batch_norm, n_resnetblocks, AVP=args.AVP)
    if "CCTA" in args.data_dir:
        validation_images, validation_landmarks, bns = data.loadCCTA(args.data_dir + "/" + dataset, args.voxel_size)
        vs = (args.voxel_size, args.voxel_size, args.voxel_size)
    elif 'Bulbus' in args.data_dir:
        validation_images, validation_landmarks, bns = data.loadBulbusMRI(args.data_dir + "/" + dataset, rf_list[-1, 0])
        vs = (0.46875, 0.46875, 0.46875)

    if single_lm:
        validation_landmarks = validation_landmarks[:, args.lm]
        validation_landmarks = validation_landmarks[:, np.newaxis, :]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)
    network.eval()

    if task == "validation":
        choosen_iter="ALL"
        average_errors = np.empty((29, n_class))
        median_errors = np.empty((29, n_class))
        #test network for x iterations on validationset
        for iter in range(20000, 300000, 10000):
            print("Iteration: ", str(iter))
            w = exp_dir + '/model_'+ str(iter) + '.pt'
            # if iter == 200000:
            #     w = exp_dir / 'model_'+ str(iter) + '_final.pt'
            network.load_state_dict(torch.load(w)['model'])
            torch.no_grad()
            times, distance_errors = test_network(network, rf_list, validation_images, validation_landmarks, heatmap, gt, vs, n_class, bns)
            for i in range(n_class):
                average_errors[(iter // 10000 - 2), i] = np.average(distance_errors[:, i])
                median_errors[(iter // 10000 - 2), i] = np.median(distance_errors[:, i])
            create_boxplot(distance_errors, save_dir + "_boxplot_"+ task + "_" + dataset + "_" + str(iter))
        df = pandas.DataFrame(average_errors)
        df.to_excel(save_dir + "_averageErrors_" + task + "_" + dataset + ".xlsx", index=False)
        df = pandas.DataFrame(median_errors)
        df.to_excel(save_dir + "_medianErrors_" + task + "_" + dataset + ".xlsx", index=False)


    elif "testing" in task:
        w = exp_dir + '/model_' + str(choosen_iter) + '.pt'
        # if choosen_iter == 200000:
        #     w = exp_dir + '/model_' + str(choosen_iter) + '_final.pt'
        network.load_state_dict(torch.load(w)['model'])
        torch.no_grad()
        times, distance_errors = test_network(network, rf_list, validation_images, validation_landmarks, heatmap, gt, vs, n_class, bns)
        1/0
        df = pandas.DataFrame(distance_errors)
        df.to_excel(save_dir + "_DistanceErrors_" + task + "_" + dataset + "_" + str(choosen_iter) + ".xlsx", index=False)
        create_boxplot(distance_errors, save_dir + "_boxplot_"+ task + "_" + dataset + "_" + str(choosen_iter))
        df = pandas.DataFrame(times)
        df.to_excel(save_dir + "_TIMES_" + task + "_" + dataset + "_" + str(choosen_iter) + ".xlsx",
                    index=False)


    df = pandas.DataFrame(bns)
    df.to_excel(save_dir + "_BNS_" + task + ".xlsx", index=False)


if __name__ == '__main__':
    rs = np.random.RandomState(123)
    args_inp = parse_args()
    save_dir = r"/home/julia/landmark_detection/rebuttal_final/results/" + args_inp.experiment_directory.split("/")[-1]
    # main(task="validation", dataset = "val", choosen_iter=args_inp.choosen_iter, heatmap=args_inp.heatmap)
    main(task="testing", dataset = "test", choosen_iter=args_inp.choosen_iter, heatmap=args_inp.heatmap)
    main(task="testing", dataset = "val", choosen_iter=args_inp.choosen_iter, heatmap=args_inp.heatmap)
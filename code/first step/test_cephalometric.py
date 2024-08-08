from __future__ import print_function
from networks_2D import Network
import torch
import numpy as np
import math
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

def loadExperimentSettings(fname):
    with open(fname, 'r') as fp:
        args = argparse.Namespace(**yaml.load(fp))
    return args

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation network')
    parser.add_argument('--experiment_directory', type=str, help='directory for experiment outputs')
    parser.add_argument('--chosen_iter', type=int, default=0)
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

def test_network(network, rf_list, images, landmarks, heatmap, gt, bns):
    times = []
    distance_errors = np.empty((len(images), n_class))
    vs=(0.1,0.1)
    for imidx in range(len(images)):
        print("Image index: ", str(imidx))
        im = images[imidx]
        lms = landmarks[imidx]
        filename = bns[imidx]
        y_padding = rf_list[-1, 1] - (im.shape[0]%rf_list[-1, 1])
        x_padding = rf_list[-1, 1] - (im.shape[1]%rf_list[-1, 1])

        if y_padding == rf_list[-1,1]:
            y_padding = 0
        if x_padding == rf_list[-1,1]:
            x_padding = 0
        im = np.pad(im, ((0, y_padding), (0, x_padding)), mode="constant")
        centerpoints = computeCenterpoints(im.shape, rf_list[:, 1])

        start = time.time()
        output = network(torch.from_numpy(im[None, None]).type(torch.FloatTensor).cuda())


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
                pred_y_loc = np.average(locs[0].flatten(), weights=pred_class[c].flatten())
                pred_x_loc = np.average(locs[1].flatten(), weights=pred_class[c].flatten())

                averaged_y_overresults[c] += pred_y_loc
                averaged_x_overresults[c] += pred_x_loc

                # if heatmap: #make and save heatmaps
                #     makeheatmaps(im, lms[c], real_pred_dist[c] + centerpoints[out_idx],
                #              pred_class[c], str(imidx)+"_lm" + str(c), save_dir)
            # if heatmap:  # make and save heatmaps
            #     makeheatmaps_mulitlm(im, lms, real_pred_dist + centerpoints[out_idx],
            #                  pred_class, str(imidx) + "_lm" + "result_" + str(out_idx), save_dir)

        predlms = []
        for c_idx in range(n_class):
            pred_y = int((averaged_y_overresults[c_idx]/len(output)) + 0.5)
            pred_x = int((averaged_x_overresults[c_idx]/len(output)) + 0.5)
            lm = lms[c_idx]

            y_error = (pred_y-lm[0])*vs[0]
            x_error = (pred_x-lm[1])*vs[1]

            distance_errors[imidx, c_idx]= math.sqrt(((y_error**2)+(x_error**2)))
            # print("Predicted lm: ", pred_y, pred_x, " Reference lm: ", lm)
            # print("Error: ", distance_errors[imidx, c_idx])
            # predlms.append((pred_y, pred_x))
        times.append(time.time() - start)
    return times, distance_errors



def main(task, dataset, chosen_iter, heatmap):
    exp_dir = args_inp.experiment_directory
    args = loadExperimentSettings(exp_dir + '/settings.yaml')
    exp_name = exp_dir.split("/")[-1]

    if args.regr_trans == 'log':
        gt = 10
    elif args.regr_trans == 'ln':
        gt = np.e

    #load data
    validation_images, validation_landmarks, bns = data.loadCephalometric(args.data_dir + "/" + dataset)
    network, rf_list = Network(args.network, 19, args.n_convpairs, args.n_downsampling, args.interlayers,
                               args.prod_class, args.sevxsev, args.batch_norm, args.AVP)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)
    network.eval()

    if task == "validation":
        chosen_iter="ALL"
        average_errors = np.empty((29, n_class))
        median_errors = np.empty((29, n_class))
        #test network for x iterations on validationset
        for iter in range(20000, 310000, 10000):
            print("Iteration: ", str(iter))
            w = exp_dir + '/model_'+ str(iter) + '.pt'
            # if iter == 200000:
            #     w = exp_dir / 'model_'+ str(iter) + '_final.pt'
            network.load_state_dict(torch.load(w)['model'])
            torch.no_grad()
            times, distance_errors = test_network(network, rf_list, validation_images, validation_landmarks, heatmap, gt, bns)
            for i in range(n_class):
                average_errors[(iter // 10000 - 2), i] = np.average(distance_errors[:, i])
                median_errors[(iter // 10000 - 2), i] = np.median(distance_errors[:, i])
            create_boxplot(distance_errors, save_dir + "_boxplot_"+ task + "_" + dataset + "_" + str(iter))
        df = pandas.DataFrame(average_errors)
        df.to_excel(save_dir + "_averageErrors_" + task + "_" + dataset + ".xlsx", index=False)
        df = pandas.DataFrame(median_errors)
        df.to_excel(save_dir + "_medianErrors_" + task + "_" + dataset + ".xlsx", index=False)


    elif "testing" in task:
        w = exp_dir + '/model_' + str(chosen_iter) + '.pt'
        # if chosen_iter == 200000:
        #     w = exp_dir + '/model_' + str(chosen_iter) + '_final.pt'
        network.load_state_dict(torch.load(w)['model'])
        torch.no_grad()
        times, distance_errors = test_network(network, rf_list, validation_images, validation_landmarks, heatmap, gt, bns)
        df = pandas.DataFrame(distance_errors)
        df.to_excel(save_dir + "_DistanceErrors_" + task + "_" + dataset + "_" + str(chosen_iter) + ".xlsx", index=False)
        df = pandas.DataFrame(times)
        df.to_excel(save_dir + "_TIMES_" + task + "_" + dataset + "_" + str(chosen_iter) + ".xlsx",
                    index=False)
        create_boxplot(distance_errors, save_dir + "_boxplot_"+ task + "_" + dataset + "_" + str(chosen_iter))

    df = pandas.DataFrame(bns)
    df.to_excel(save_dir + "_BNS_" + task + ".xlsx", index=False)


if __name__ == '__main__':
    rs = np.random.RandomState(123)
    n_class=19
    args_inp = parse_args()
    save_dir = r"/home/julia/landmark_detection/rebuttal_final/results/" + args_inp.experiment_directory.split("/")[-1]
    # main(task="validation", dataset = "val", chosen_iter=args_inp.chosen_iter, heatmap=args_inp.heatmap)
    # main(task="testing", dataset = "val", chosen_iter=args_inp.chosen_iter, heatmap=args_inp.heatmap)
    main(task="testing", dataset = "test1", chosen_iter=args_inp.chosen_iter, heatmap=args_inp.heatmap)
    main(task="testing", dataset = "test2", chosen_iter=args_inp.chosen_iter, heatmap=args_inp.heatmap)
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm
from matplotlib.colors import colorConverter, LinearSegmentedColormap, ListedColormap
import matplotlib
import cv2



def makeheatmaps_mulitlm(im,lms_all, regressed_locations_all, predicted_probabilities_all, imname, save_dir):
    x_size, y_size = im.shape

    # classification heatmap
    plt.imshow(im.T, cmap=plt.cm.gray)
    predicted_probabilities_all /= (predicted_probabilities_all.max() / 1.0)
    predicted_probabilities_all=predicted_probabilities_all>0.5
    predicted_probabilities = np.sum(predicted_probabilities_all, axis=0)

    classification_heatmap = cv2.resize(np.asarray(predicted_probabilities > 0.5, dtype=int), (y_size, x_size),
                                        interpolation=cv2.INTER_NEAREST)
    plt.scatter(lms_all[:,1], lms_all[:,0], s=1, c='red', marker='.')
    plt.imshow(classification_heatmap, interpolation='lanczos', cmap=plt.cm.viridis, alpha=.9)
    plt.savefig(save_dir + "classification_heatmap_" + imname + ".png")
    plt.close()

    # create regression heatmap
    # plt.imshow(im, cmap=plt.cm.gray)
    regression_heatmap = np.zeros(im.shape)
    regressed_locations_all[:, 0] = np.clip(regressed_locations_all[:, 0], 0, im.shape[0] - 1)
    regressed_locations_all[:, 1] = np.clip(regressed_locations_all[:, 1], 0, im.shape[1] - 1)
    for i in range(len(regressed_locations_all)):
        for x in range(regressed_locations_all.shape[2]):
            for y in range(regressed_locations_all.shape[3]):
                regression_heatmap[
                    int(regressed_locations_all[i, 0, x, y]), int(regressed_locations_all[i, 1, x, y])] += 1

    plt.imshow(regression_heatmap, interpolation='None', cmap=plt.cm.viridis, alpha=.9)
    plt.scatter(lms_all[:, 1], lms_all[:, 0], s=1, c='red', marker='.')
    plt.savefig(save_dir + "regression_heatmap_" + imname + ".png")
    plt.close()

def makeheatmaps(im,lm, regressed_locations, predicted_probabilities, imname, save_dir):
    x_size, y_size = im.shape
    # cmap2 = LinearSegmentedColormap.from_list('my_cmap2', ['black', 'green'], 256)
    # cmap2._init()
    # alphas = np.linspace(0.0, 0.5, cmap2.N + 3)
    # cmap2._lut[:, -1] = alphas

    # classification heatmap
    predicted_probabilities /= (predicted_probabilities.max() / 1.0)
    classification_heatmap = cv2.resize(np.asarray(predicted_probabilities > 0.5, dtype=int), (y_size, x_size),
                                        interpolation=cv2.INTER_NEAREST)

    plt.imshow(im.T, cmap=plt.cm.gray)
    plt.scatter(lm[1], lm[0], c='white', marker='.')
    plt.imshow(classification_heatmap, interpolation='lanczos', cmap=plt.cm.viridis, alpha=.5)
    plt.savefig(save_dir + "classification_heatmap_" + imname + ".png")
    plt.close()

    # create regression heatmap
    regressed_locations[0][regressed_locations[0] > im.shape[0]] = im.shape[0] - 1
    regressed_locations[1][regressed_locations[1] > im.shape[1]] = im.shape[1] - 1
    regression_heatmap = np.zeros(im.shape)
    for x in range(regressed_locations.shape[1]):
        for y in range(regressed_locations.shape[2]):
            regression_heatmap[int(regressed_locations[0, x, y]), int(regressed_locations[1, x, y])]+=1
    plt.imshow(im.T, cmap=plt.cm.gray)
    plt.scatter(lm[1], lm[0], c='white', marker='.')
    plt.imshow(regression_heatmap, interpolation='lanczos', cmap=plt.cm.viridis, alpha=.5)
    plt.savefig(save_dir + "regression_heatmap_" + imname + ".png")
    plt.close()

def create_boxplot(dist_errors, n):
    dist_errors=np.asarray(dist_errors)
    fig, axes = plt.subplots(figsize=(12, 16))
    bplot= axes.boxplot(dist_errors, patch_artist=True)

    # Fill with colors
    cm = plt.cm.get_cmap('rainbow')
    colors = [cm(val / len(dist_errors)) for val in range(len(dist_errors))]

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    # axes.set_ylim((0, 10))
    plt.savefig(
        n + ".png",bbox_inches='tight',
                pad_inches=0, transparent=True)
    plt.close()
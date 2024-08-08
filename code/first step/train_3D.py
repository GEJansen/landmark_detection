from networks_3D import Network
import torch
import numpy as np
import argparse
import os
import sys
import yaml
from pathlib import Path
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import data
from torchvision import transforms
from pathlib import Path
from torch.utils.data import RandomSampler, DataLoader
from tqdm import tqdm
from test_3D import computeCenterpoints, loginv
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def loadExperimentSettings(fname):
    with open(fname, 'r') as fp:
        args = argparse.Namespace(**yaml.load(fp))
    return args

def saveExperimentSettings(args, fname):
    with open(fname, 'w') as fp:
        yaml.dump(vars(args), fp)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a landmark localization 3D network')
    parser.add_argument('--output_directory', type=str, help='directory for experiment outputs')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-vs', '--voxel_size', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--chunk_size', type=int, default=64)
    parser.add_argument('--n_convpairs', type=int, default=2,
                        help='Number of conv pairs per resnetblock')
    parser.add_argument('--n_downsampling', type=int, default=5,
                        help='Number of downsampling')
    parser.add_argument('--n_resnetblocks', type=int, default=4,
                        help='Number of resnetblocks')
    parser.add_argument('--max_iters', type=int, default=300000)
    parser.add_argument('--rs', type=int, default=7)
    parser.add_argument('--batch_norm', type=int, default=1, choices=[0,1])
    parser.add_argument('--AVP', type=int, default=0, choices=[0, 1])
    parser.add_argument('--store_model_every', type=int, default=5000)
    parser.add_argument('--evaluate_model_every', type=int, default=1000)
    parser.add_argument('--regr_loss', choices=['mse', 'l1'], default='l1')
    parser.add_argument('--regr_trans', choices=['log', 'ln'], default='log')
    parser.add_argument('--masked_loss', action='store_true')
    parser.add_argument('--lr_decay', action='store_true')
    parser.add_argument('--weight_decay', action='store_true')
    parser.add_argument('--prod_class', action='store_true')
    parser.add_argument('--balance_losses', action='store_true')
    parser.add_argument('--single_lm', action='store_true')
    parser.add_argument('--lm', type=int, default=-1)
    parser.add_argument('--sevxsev', action='store_true')
    parser.add_argument('--interlayers', nargs='+', type=int,
                        help='Ints indicating after which blocks you want deep supervision. Example: 0 1 2', default=None)
    parser.add_argument('--network', type=str,
                        choices=['resnet', 'resnet_deepsupervision', 'resnet_34_CCTA05', 'resnet_34_BULB',
                                 'resnet_34_BULB_big', 'resnet_34_BULB_48C', 'resnet_34_BULB_64C', 'resnet_34_BULB_96C',
                                 'resnet_34_real', 'resnet_34_real_equalsplit', 'resnet_34_real_unevensplit'],
                        default='resnet_34_real')
    parser.add_argument('--data_dir', type=str, default='/home/julia/landmark_detection/paper/data/images/')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.rs)  # to be able to reproduce results
    torch.manual_seed(args.rs)
    dst_dir = Path(args.output_directory)
    deep_supervision = 'deepsupervision' in args.network

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        writer = SummaryWriter(dst_dir)
    else:
        print("Warning: you have already run a network with the same parameters. "
              "If you want to run it again, move or delete the previous results "
              "that are stored here:\n%s" % dst_dir)
        sys.exit()
    saveExperimentSettings(args, dst_dir / 'settings.yaml')


    if 'CCTA' in args.data_dir:
        n_class=8
        lms_interest = [1, 4, 6]
    elif 'Bulbus' in args.data_dir:
        n_class=2
        lms_interest = [0,1]
    if args.single_lm:
        n_class=1
        lms_interest=[args.lm]

    network, rf_list = Network(args.network, n_class, args.n_convpairs, args.n_downsampling, args.interlayers,
                               args.prod_class, args.sevxsev, args.batch_norm, args.n_resnetblocks, args.chunk_size, args.AVP)

    #patchsize, chunksize, landmark grids
    ps = rf_list[-1,1]
    chunk_size = (ps * (rf_list[-1,0]//ps+1),
                  ps * (rf_list[-1,0]//ps+1),
                  ps * (rf_list[-1,0]//ps+1))
    landmark_grids = []
    for r in range(len(rf_list)):
        rf_size = np.array((rf_list[r,1], rf_list[r,1], rf_list[r,1]))
        landmark_grid = chunk_size // rf_size
        landmark_grids.append(landmark_grid)

    #Loss functions
    class_loss = torch.nn.BCELoss()
    regr_loss = torch.nn.L1Loss()
    if args.regr_loss == "mse":
        regr_loss = torch.nn.MSELoss() #mean squared error
    regr_trans=np.log10
    gt = 10
    if args.regr_trans == 'ln':
        regr_trans=np.log
        gt = np.e

    #Make and save network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)
    optimizer = optim.Adam(network.parameters(), lr=args.learning_rate, amsgrad=True)
    if args.weight_decay:
        optimizer = optim.Adam(network.parameters(), lr=args.learning_rate, amsgrad=True, weight_decay=0.00005)
    if args.lr_decay:
        milestones = []
        for i in range(100000, args.max_iters, 50000):
            milestones.append(i)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=milestones, gamma=0.1)
    else:
        scheduler = scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[args.max_iters], gamma=0.1) #no learning rate scheduling

    fname = '{}/model_{}.pt'.format(args.output_directory, 0)
    torch.save({'model': network.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()},
               fname)

    #Load Data
    if "CCTA" in args.data_dir:
        training_images, training_landmarks, _ = data.loadCCTA(args.data_dir + "/train", args.voxel_size)
        validation_images, validation_landmarks, _ = data.loadCCTA(args.data_dir + "/val", args.voxel_size)
    elif 'Bulbus' in args.data_dir:
        training_images, training_landmarks, _ = data.loadBulbusMRI(args.data_dir + "/train", rf_list[-1, 0])
        validation_images, validation_landmarks, _ = data.loadBulbusMRI(args.data_dir + "/val", rf_list[-1, 0])
    if args.single_lm:
        training_landmarks=training_landmarks[:,args.lm]
        training_landmarks=training_landmarks[:, np.newaxis, :]
        validation_landmarks=validation_landmarks[:,args.lm]
        validation_landmarks = validation_landmarks[:, np.newaxis, :]


    #batch iterators
    batch_iter_training = data.BatchIterator_3D(training_images, training_landmarks, args.batch_size, chunk_size,
                                                n_class, rf_list, landmark_grids, regr_trans)
    batch_iter_validation = data.BatchIterator_3D(validation_images, validation_landmarks, args.batch_size, chunk_size,
                                                  n_class, rf_list, landmark_grids, regr_trans)
    alpha=1.0
    if args.balance_losses:
        if 'CCTA' in args.data_dir:
            alpha = 8.0
        elif 'Bulbus' in args.data_dir:
            alpha = 2.0
    try:
        for iter, training_batch in tqdm(enumerate(batch_iter_training), total=args.max_iters):

            ims, dist, labs = training_batch
            ims = torch.from_numpy(ims).to('cuda')
            optimizer.zero_grad()
            intermediate_output = network(ims)

            r_l = regr_loss(intermediate_output[-1][1],
                            torch.from_numpy(np.asarray(dist[-1])).type(torch.FloatTensor).to('cuda'))

            c_l = class_loss(intermediate_output[-1][0],
                             torch.from_numpy(np.asarray(labs[-1])).type(torch.FloatTensor).to('cuda'))*alpha

            iregr_loss = 0
            iclass_loss = 0
            for i in range(len(intermediate_output) - 2, -1, -1):
                iregr_loss += regr_loss(intermediate_output[i][1],
                                        torch.from_numpy(np.asarray(dist[i])).type(torch.FloatTensor).to('cuda'))
                iclass_loss += class_loss(intermediate_output[i][0],
                                   torch.from_numpy(np.asarray(labs[i])).type(torch.FloatTensor).to('cuda'))*alpha


            loss = r_l + c_l + iregr_loss + iclass_loss
            loss.backward()
            optimizer.step()

            if args.lr_decay:
                scheduler.step()

            if not iter % args.evaluate_model_every:
                print("Loss: ", loss.item())
                with torch.no_grad():
                    network.eval()
                    ims_val, dist_val, labs_val = next(batch_iter_validation)
                    ims_val = torch.from_numpy(ims_val).to('cuda')
                    intermediate_output_val = network(ims_val)
                    network.train()

                    rv_l = regr_loss(intermediate_output_val[-1][1],
                                     torch.from_numpy(np.asarray(dist_val[-1])).type(torch.FloatTensor).to('cuda'))

                    cv_l = class_loss(intermediate_output_val[-1][0],
                                      torch.from_numpy(np.asarray(labs_val[-1])).type(torch.FloatTensor).to('cuda'))*alpha

                    ivregr_loss = 0
                    ivclass_loss = 0
                    for i in range(len(intermediate_output) - 2, -1, -1):
                        ivregr_loss += regr_loss(intermediate_output_val[i][1],
                                                 torch.from_numpy(np.asarray(dist_val[i])).type(torch.FloatTensor).to(
                                                     'cuda'))
                        ivclass_loss += (class_loss(intermediate_output_val[i][0],
                                                    torch.from_numpy(np.asarray(labs_val[i])).type(torch.FloatTensor).to(
                                                        'cuda')))*alpha
                    loss_val = rv_l + cv_l + ivregr_loss + ivclass_loss
                    writer.add_scalar('Validation/classification', cv_l.detach().cpu().numpy(), iter)
                    writer.add_scalar('Validation/regression', rv_l.detach().cpu().numpy(), iter)
                    writer.add_scalar('Validation/total', loss_val.detach().cpu().numpy(), iter)
                    if deep_supervision:
                        writer.add_scalar('Validation/deep_classification', ivclass_loss.detach().cpu().numpy(), iter)
                        writer.add_scalar('Validation/deep_regression', ivregr_loss.detach().cpu().numpy(), iter)

                    # evaluate 3 images:
                    network.eval()
                    display_ims = []
                    for i in range(3):
                        ims_val = validation_images[i]
                        norm = np.linalg.norm(ims_val)
                        ims_val = (ims_val / norm) * 256
                        ims_val = np.pad(ims_val, 30, mode='constant')
                        z_padding = rf_list[-1, 1] - (ims_val.shape[0] % rf_list[-1, 1])
                        y_padding = rf_list[-1, 1] - (ims_val.shape[1] % rf_list[-1, 1])
                        x_padding = rf_list[-1, 1] - (ims_val.shape[2] % rf_list[-1, 1])

                        if z_padding == rf_list[-1, 1]:
                            z_padding = 0
                        if y_padding == rf_list[-1, 1]:
                            y_padding = 0
                        if x_padding == rf_list[-1, 1]:
                            x_padding = 0
                        ims_valwhole = np.pad(ims_val, ((0, z_padding), (0, y_padding), (0, x_padding)), mode="constant")
                        ims_lms_all = validation_landmarks[i]+30
                        centerpoints = computeCenterpoints(ims_valwhole.shape, rf_list[:, 1])
                        output = network(torch.from_numpy(ims_valwhole[None, None]).type(torch.FloatTensor).cuda())
                        pred_class = output[0][0][0].cpu().detach().numpy()
                        log_pred_dist = output[0][1][0].cpu().detach().numpy()
                        real_pred_dist = loginv(log_pred_dist, gt)
                        for i in range(len(lms_interest)):
                            n = lms_interest[i]
                            if args.single_lm:
                                ims_lms= np.asarray(ims_lms_all[0], dtype=int)
                                n = 0
                            else:
                                ims_lms = np.asarray(ims_lms_all[n], dtype=int)
                            locs = real_pred_dist[n] + centerpoints[0]
                            pred_z_loc = np.average(locs[0].flatten(), weights=pred_class[n].flatten())
                            pred_z_loc = int(pred_z_loc-(ims_lms[0]-30))
                            pred_y_loc = np.average(locs[1].flatten(), weights=pred_class[n].flatten())
                            pred_y_loc = int(pred_y_loc-(ims_lms[1]-30))
                            pred_x_loc = np.average(locs[2].flatten(), weights=pred_class[n].flatten())
                            pred_x_loc = int(pred_x_loc-(ims_lms[1]-30))

                            ims_valx = ims_valwhole[ims_lms[0], ims_lms[1]-30:ims_lms[1]+30,ims_lms[2]-30:ims_lms[2]+30]
                            ims_valy = ims_valwhole[ims_lms[0]-30:ims_lms[0]+30, ims_lms[1],ims_lms[2]-30:ims_lms[2]+30]
                            ims_valz = ims_valwhole[ims_lms[0]-30:ims_lms[0]+30,ims_lms[1]-30:ims_lms[1]+30,ims_lms[2]]

                            fig, ax = plt.subplots(1, 3)
                            ax[0].imshow(ims_valx, cmap=plt.cm.gray)
                            ax[0].scatter(pred_y_loc, pred_x_loc, c='r')
                            ax[0].scatter(30,30, c='g')
                            ax[1].imshow(ims_valy, cmap=plt.cm.gray)
                            ax[1].scatter(pred_z_loc, pred_x_loc, c='r')
                            ax[1].scatter(30, 30, c='g')
                            ax[2].imshow(ims_valz, cmap=plt.cm.gray)
                            ax[2].scatter(pred_z_loc, pred_y_loc, c='r')
                            ax[2].scatter(30, 30, c='g')
                            [axi.set_axis_off() for axi in ax.ravel()]
                            display_ims.append(fig)
                    writer.add_figure("3 validation images 3x2D", display_ims, iter)
                    network.train()

            if not iter % args.store_model_every:
                fname = '{}/model_{}.pt'.format(args.output_directory, iter)
                torch.save({'model': network.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict()},
                           fname)

            writer.add_scalar('Training/classification', c_l.detach().cpu().numpy(), iter)
            writer.add_scalar('Training/regression', r_l.detach().cpu().numpy(), iter)
            writer.add_scalar('Training/total', loss.detach().cpu().numpy(), iter)

            if deep_supervision:
                writer.add_scalar('Training/deep_regression', iregr_loss.detach().cpu().numpy(), iter)
                writer.add_scalar('Training/deep_classification', iclass_loss.detach().cpu().numpy(), iter)

            if iter >= args.max_iters:
                break
    except KeyboardInterrupt:
        print('interrupted')
        fname = '{}model_{}_interrupted.pt'.format(args.output_directory, iter)
        torch.save({'model': network.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()},
                   fname)

    finally:
        fname = '{}/model_{}_final.pt'.format(args.output_directory, args.max_iters)
        torch.save({'model': network.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()},
                   fname)


if __name__ == '__main__':
    main()
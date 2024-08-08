from networks_2D import Network
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

def loadExperimentSettings(fname):
    with open(fname, 'r') as fp:
        args = argparse.Namespace(**yaml.load(fp))
    return args

def saveExperimentSettings(args, fname):
    with open(fname, 'w') as fp:
        yaml.dump(vars(args), fp)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation for Cephalometric X-rays network')
    parser.add_argument('--output_directory', type=str, help='directory for experiment outputs')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_convpairs', type=int, default=2,
                        help='Number of conv pairs per resnetblock')
    parser.add_argument('--n_downsampling', type=int, default=5,
                        help='Number of resnet blocks')
    parser.add_argument('--max_iters', type=int, default=300000)
    parser.add_argument('--rs', type=int, default=7)
    parser.add_argument('--batch_norm', type=int, default=1, choices=[0,1])
    parser.add_argument('--AVP', type=int, default=0, choices=[0,1])
    parser.add_argument('--store_model_every', type=int, default=5000)
    parser.add_argument('--evaluate_model_every', type=int, default=1000)
    parser.add_argument('--regr_loss', choices=['mse', 'l1'], default='l1')
    parser.add_argument('--regr_trans', choices=['log', 'ln'], default='log')
    parser.add_argument('--masked_loss', action='store_true')
    parser.add_argument('--lr_decay', action='store_true')
    parser.add_argument('--weight_decay', action='store_true')
    parser.add_argument('--prod_class', action='store_true')
    parser.add_argument('--balance_losses', action='store_true')
    parser.add_argument('--sevxsev', action='store_true')
    parser.add_argument('--interlayers', nargs='+', type=int,
                        help='Ints indicating after which blocks you want deep supervision. Example: 0 1 2', default=None)
    parser.add_argument('--network', type=str, choices=['resnet', 'resnet_deepsupervision', 'resnet_34', 'resnet_50'], default='resnet')
    parser.add_argument('--data_dir', type=str, default='/home/julia/landmark_detection/paper/data/images/Cephalometric')
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

    network, rf_list = Network(args.network, 19, args.n_convpairs, args.n_downsampling, args.interlayers,
                               args.prod_class, args.sevxsev, args.batch_norm, args.AVP)

    #patchsize, chunksize, landmark grids
    ps = 2 ** args.n_downsampling
    chunk_size = (ps * (rf_list[-1,0]//ps+1),
                  ps * (rf_list[-1,0]//ps+1))
    print(chunk_size)
    landmark_grids = []
    for r in range(len(rf_list)):
        rf_size = np.array((rf_list[r,1], rf_list[r,1]))
        landmark_grid = chunk_size // rf_size
        landmark_grids.append(landmark_grid)

    #Loss functions
    class_loss = torch.nn.BCELoss()
    regr_loss = torch.nn.L1Loss()
    if args.regr_loss == "mse":
        regr_loss = torch.nn.MSELoss() #mean squared error
    regr_trans=np.log10
    if args.regr_trans == 'ln':
        regr_trans=np.log

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
    training_images, training_landmarks, _ = data.loadCephalometric(args.data_dir + "/train")
    validation_images, validation_landmarks, _ = data.loadCephalometric(args.data_dir + "/val")

    #batch iterators
    batch_iter_training = data.BatchIterator_2D(training_images, training_landmarks, args.batch_size, chunk_size,
                                                19, rf_list, landmark_grids, regr_trans)
    batch_iter_validation = data.BatchIterator_2D(validation_images, validation_landmarks, args.batch_size, chunk_size,
                                                  19, rf_list, landmark_grids, regr_trans)
    alpha=1.0
    if args.balance_losses:
        alpha = 20
    try:
        for iter, training_batch in tqdm(enumerate(batch_iter_training), total=args.max_iters):

            ims, dist, labs = training_batch
            ims = torch.from_numpy(ims).to('cuda')
            optimizer.zero_grad()
            intermediate_output = network(ims)
            masked_regr_loss=0
            if args.masked_loss: #mask regressionloss
                masked_labs = np.repeat(np.asarray(labs[-1])[:,:,np.newaxis, :, :], 2, axis=2)
                masked_dist = np.asarray(dist[-1]) * masked_labs
                masked_dist = torch.from_numpy(masked_dist).type(torch.FloatTensor).to('cuda')
                masked_output = intermediate_output[-1][1].detach().cpu().numpy() * masked_labs
                masked_output = torch.from_numpy(masked_output).type(torch.FloatTensor).to('cuda')
                masked_regr_loss= regr_loss(masked_output, masked_dist)*1000

            r_l = regr_loss(intermediate_output[-1][1],
                            torch.from_numpy(np.asarray(dist[-1])).type(torch.FloatTensor).to('cuda'))+masked_regr_loss

            c_l = class_loss(intermediate_output[-1][0],
                             torch.from_numpy(np.asarray(labs[-1])).type(torch.FloatTensor).to('cuda'))*alpha

            iregr_loss = 0
            iclass_loss = 0
            for i in range(len(intermediate_output) - 2, -1, -1):
                imasked_regr_loss = 0
                if args.masked_loss:  # mask regressionloss
                    masked_labs = np.repeat(np.asarray(labs[i])[:, :, np.newaxis, :, :], 2, axis=2)
                    masked_dist = np.asarray(dist[i]) * masked_labs
                    masked_dist = torch.from_numpy(masked_dist).type(torch.FloatTensor).to('cuda')
                    masked_output = intermediate_output[i][1].detach().cpu().numpy() * masked_labs
                    masked_output = torch.from_numpy(masked_output).type(torch.FloatTensor).to('cuda')
                    imasked_regr_loss = regr_loss(masked_output, masked_dist) * 1000
                iregr_loss += regr_loss(intermediate_output[i][1],
                                        torch.from_numpy(np.asarray(dist[i])).type(torch.FloatTensor).to('cuda')) + imasked_regr_loss
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

                    vmasked_regr_loss = 0
                    if args.masked_loss:  # mask regressionloss
                        masked_labs = np.repeat(np.asarray(labs_val[-1])[:, :, np.newaxis, :, :], 2, axis=2)
                        masked_dist = np.asarray(dist_val[-1]) * masked_labs
                        masked_dist = torch.from_numpy(masked_dist).type(torch.FloatTensor).to('cuda')
                        masked_output = intermediate_output_val[-1][1].detach().cpu().numpy() * masked_labs
                        masked_output = torch.from_numpy(masked_output).type(torch.FloatTensor).to('cuda')
                        vmasked_regr_loss = regr_loss(masked_output, masked_dist) * 1000

                    rv_l = regr_loss(intermediate_output_val[-1][1],
                                     torch.from_numpy(np.asarray(dist_val[-1])).type(torch.FloatTensor).to('cuda')) + vmasked_regr_loss

                    cv_l = class_loss(intermediate_output_val[-1][0],
                                      torch.from_numpy(np.asarray(labs_val[-1])).type(torch.FloatTensor).to('cuda'))*alpha

                    ivregr_loss = 0
                    ivclass_loss = 0
                    for i in range(len(intermediate_output) - 2, -1, -1):
                        ivmasked_regr_loss = 0
                        if args.masked_loss:  # mask regressionloss
                            masked_labs = np.repeat(np.asarray(labs_val[i])[:, :, np.newaxis, :, :], 2, axis=2)
                            masked_dist = np.asarray(dist_val[i]) * masked_labs
                            masked_dist = torch.from_numpy(masked_dist).type(torch.FloatTensor).to('cuda')
                            masked_output = intermediate_output_val[i][1].detach().cpu().numpy() * masked_labs
                            masked_output = torch.from_numpy(masked_output).type(torch.FloatTensor).to('cuda')
                            ivmasked_regr_loss = regr_loss(masked_output, masked_dist) * 1000

                        ivregr_loss += regr_loss(intermediate_output_val[i][1],
                                                 torch.from_numpy(np.asarray(dist_val[i])).type(torch.FloatTensor).to(
                                                     'cuda'))+ivmasked_regr_loss
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
from networks_2D import MultiFineTune
import torch
import numpy as np
import argparse
import os
import sys
import yaml
from pathlib import Path
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import data_augmentation_2D as data
from torchvision import transforms
from pathlib import Path
from torch.utils.data import RandomSampler, DataLoader
from tqdm import tqdm
from test_network import computeCenterpoints, loginv
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def loadExperimentSettings(fname):
    with open(fname, 'r') as fp:
        args = argparse.Namespace(**yaml.load(fp))
    return args

def saveExperimentSettings(args, fname):
    with open(fname, 'w') as fp:
        yaml.dump(vars(args), fp)


def determine_distances(lms_a, lms_b):
    return np.sqrt((lms_a - lms_b)**2).sum(1)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a landmark localization 2D network')
    parser.add_argument('--output_directory', type=str, help='directory for experiment outputs')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-vs', '--voxel_size', type=float, default=1.5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_convpairs', type=int, default=2,
                        help='Number of conv pairs per resnetblock')
    parser.add_argument('--n_downsampling', type=int, default=2,
                        help='Number of downsampling')
    parser.add_argument('--n_resnetblocks', type=int, default=4,
                        help='Number of resnetblocks')
    parser.add_argument('--max_iters', type=int, default=310000)
    parser.add_argument('--rs', type=int, default=7)
    parser.add_argument('--batch_norm', type=int, default=1, choices=[0,1])
    parser.add_argument('--store_model_every', type=int, default=5000)
    parser.add_argument('--evaluate_model_every', type=int, default=1000)
    parser.add_argument('--masked_loss', action='store_true')
    parser.add_argument('--lr_decay', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--prod_class', action='store_true')
    parser.add_argument('--balance_losses', action='store_true')
    parser.add_argument('--log_loss', action='store_true')

    parser.add_argument('--data_augmentation', action='store_true')
    parser.add_argument('--lm', type=int, default=-1)
    parser.add_argument('--sevxsev', action='store_true')
    parser.add_argument('--interlayers', nargs='+', type=int,
                        help='Ints indicating after which blocks you want deep supervision. Example: 0 1 2', default=None)
    parser.add_argument('--model_file', type=str, default=None)

    parser.add_argument('--network', type=str,
                        choices=['onlyConv', 'AVPooling', 'Strided', 'RandC', 'ResnetRandC', 'ResnetRandCStrided'],
                        default='onlyConv')
    parser.add_argument('--data_dir', type=str, default='/home/julia/landmark_detection/paper/data/images/')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def main():
    print('main')
    args = parse_args()
    np.random.seed(args.rs)  # to be able to reproduce results
    torch.manual_seed(args.rs)
    dst_dir = Path(args.output_directory)

    RandC = 'RandC' in args.network #so not only regression but also classification

    if args.log_loss:
        logfn = np.log10
    else:
        logfn = None

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        writer = SummaryWriter(dst_dir)
    else:
        print("Warning: you have already run a network with the same parameters. "
              "If you want to run it again, move or delete the previous results "
              "that are stored here:\n%s" % dst_dir)
        sys.exit()
    saveExperimentSettings(args, dst_dir / 'settings.yaml')

    n_class=19

    if args.lm >= 0:
        n_class=1
        lms_interest=[args.lm]


    model = MultiFineTune(n_class, network_arch=args.network)

    #print(rf_list)
    #model = drn_d_56(num_channels=1, num_classes=n_class, pool_size=1)
    device = 'cuda'
    print('model to cuda')
    model = model.to(device)


    #patchsize, chunksize, landmark grids

    patch_size = 16


    classification_criterion = torch.nn.BCELoss()
    regression_criterion = torch.nn.L1Loss()
    #regression_criterion = torch.nn.MSELoss()

    # log_fn = np.log10
    # log_inv = lambda x: 10 ** x

    #log_fn = np.log
    #log_inv = np.exp
    #log_fn = lambda x: x


    #Make and save network

    print('optimizer')
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(model.parameters(), momentum=0.99, lr=0.000000005, nesterov=True)


    #milestones = list(range(50000, args.max_iters, 50000))
    milestones = [100000000]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.25)
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.1, step_size_up=5000, cycle_momentum=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10000, T_mult=2)

    if args.model_file:
        state_dicts = torch.load(args.model_file)
        #it = state_dicts['it']
        model.load_state_dict(state_dicts['model'])
        optimizer.load_state_dict(state_dicts['optimizer'])
        scheduler.load_state_dict(state_dicts['scheduler'])

    if RandC:
        transforms_training = transforms.Compose(
            [data.RandomSubImagesAroundLandmarks(size=patch_size, rs=np.random.RandomState(808)),
             data.ConvertToTrainingSample2D(2, logfn)])
        transforms_validation = transforms.Compose(
            [data.RandomSubImagesAroundLandmarks(size=patch_size, rs=np.random.RandomState(808)),
             data.ConvertToTrainingSample2D(2, logfn)])
    else:
        transforms_training = transforms.Compose([data.RandomSubImagesAroundLandmarks(size=patch_size, rs=np.random.RandomState(808))])
        transforms_validation = transforms.Compose([data.RandomSubImagesAroundLandmarks(size=patch_size, rs=np.random.RandomState(808))])

    print('#Load Data')
    training_set = data.CephalometricDataset('train', transform=transforms_training, single = False, lm_idx=args.lm)
    validation_set = data.CephalometricDataset('val', transform=transforms_validation, single = False, lm_idx=args.lm)

    #batch iterators
    sampler_training = RandomSampler(training_set, replacement=True, num_samples=args.batch_size * args.max_iters)

    sampler_validation = RandomSampler(validation_set, replacement=True, num_samples=args.batch_size * args.max_iters)

    data_loader_training = DataLoader(training_set,
                                      batch_size=args.batch_size,
                                      sampler=sampler_training,
                                      num_workers=2)

    data_loader_validation = DataLoader(validation_set,
                                        batch_size=args.batch_size,
                                        sampler=sampler_validation,
                                        num_workers=2)
    alpha=10
    print('start')
    try:
        for it, training_batch in tqdm(enumerate(data_loader_training), total=args.max_iters):
            ims = training_batch['image'].type(torch.FloatTensor).cuda()
            displacement = training_batch['displacement'].cuda()

            optimizer.zero_grad()
            if RandC:
                prediction_cls, prediction_rgr = model(ims)
                classes = training_batch['classes'].cuda()
                classification_loss = classification_criterion(prediction_cls, classes)
            else:
                prediction_rgr = model(ims)
                classification_loss=0

            regression_loss = regression_criterion(prediction_rgr, displacement)

            loss = regression_loss + classification_loss

            #print(loss.detach().cpu().numpy())

            if RandC:
                writer.add_scalar('Training/finetune_classification', classification_loss.detach().cpu().numpy(), it)
            writer.add_scalar('Training/finetune_regression', regression_loss.detach().cpu().numpy(), it)
            writer.add_scalar('Training/finetune', loss.detach().cpu().numpy(), it)
            #writer.add_scalar('Training/learning rate', scheduler.get_lr().item(), it)

            loss.backward()
            optimizer.step()
            scheduler.step()

            if not it % args.evaluate_model_every:
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.cpu().detach().numpy(), it)

            if not it % args.evaluate_model_every:
                print("Loss: ", loss.item())
                with torch.no_grad():
                    model.eval()
                    validaton_batch = next(iter(data_loader_validation))
                    ims_val = validaton_batch['image'].type(torch.FloatTensor).cuda()
                    dist_val = validaton_batch['displacement']

                    if RandC:
                        prediction_cls, prediction_rgr = model(ims_val)
                        classes = validaton_batch['classes'].cuda()
                        classification_loss_val = classification_criterion(prediction_cls, classes)
                    else:
                        prediction_rgr = model(ims_val)
                        classification_loss_val = 0


                    regression_loss_val = regression_criterion(prediction_rgr,
                                     torch.from_numpy(np.asarray(dist_val)).type(torch.FloatTensor).to('cuda'))

                    loss_val = regression_loss_val + classification_loss_val

                    if RandC:
                        writer.add_scalar('Validation/finetune_classification', classification_loss_val.detach().cpu().numpy(), it)
                    writer.add_scalar('Validation/finetune_regression', regression_loss_val.detach().cpu().numpy(), it)
                    writer.add_scalar('Validation/finetune', loss_val.detach().cpu().numpy(), it)

                    model.train()

            if not it % args.store_model_every:
                fname = '{}/model_{}.pt'.format(args.output_directory, it)
                torch.save({'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'iteration': it},
                           fname)




            if it >= args.max_iters:
                break
    except KeyboardInterrupt:
        print('interrupted')
        fname = '{}/model_{}_interrupted.pt'.format(args.output_directory, it)
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it},
                   fname)

    finally:
        fname = '{}/model_{}_final.pt'.format(args.output_directory, args.max_iters)
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                            'iteration': it},
                           fname)


if __name__ == '__main__':
    main()
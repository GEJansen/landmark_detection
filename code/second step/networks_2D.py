import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import nn



class FinalFineTune_onlyConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        convs = list()
        convs.append(torch.nn.Conv2d(1, 16, 3))
        convs.append(torch.nn.BatchNorm2d(16))
        convs.append(torch.nn.ReLU(inplace=True))
        for idx in range(6):
            convs.append(torch.nn.Conv2d(16, 16, 3))
            convs.append(torch.nn.BatchNorm2d(16))
            convs.append(torch.nn.ReLU(inplace=True))
        self.feats = torch.nn.Sequential(*convs)
        self.regress = torch.nn.Sequential(torch.nn.Linear(128, 64),
                                           torch.nn.ReLU(inplace=True),
                                           torch.nn.Linear(64, 64),
                                           torch.nn.ReLU(inplace=True),
                                           torch.nn.Linear(64, 3))

    def forward(self, x):
        x = self.feats(x)
        x = x.view(-1, 128)
        return self.regress(x).view(-1, 2)


class FinalFineTune_AVPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        convs = list()
        convs.append(torch.nn.Conv2d(1, 16, 3))
        convs.append(torch.nn.BatchNorm2d(16))
        convs.append(torch.nn.ReLU(inplace=True))

        convs.append(torch.nn.Conv2d(16, 16, 3))
        convs.append(torch.nn.BatchNorm2d(16))
        convs.append(torch.nn.ReLU(inplace=True))
        convs.append(torch.nn.AvgPool2d(kernel_size=2))
        for idx in range(2):
            convs.append(torch.nn.Conv2d(16, 16, 3))
            convs.append(torch.nn.BatchNorm2d(16))
            convs.append(torch.nn.ReLU(inplace=True))

        self.feats = torch.nn.Sequential(*convs)
        self.regress = torch.nn.Sequential(torch.nn.Linear(128, 64),
                                           torch.nn.ReLU(inplace=True),
                                           torch.nn.Linear(64, 64),
                                           torch.nn.ReLU(inplace=True),
                                           torch.nn.Linear(64, 3))

    def forward(self, x):
        x = self.feats(x)
        x = x.view(-1, 128)
        return self.regress(x).view(-1, 2)

class FinalFineTune_strided(torch.nn.Module):
    def __init__(self):
        super().__init__()
        convs = list()
        convs.append(torch.nn.Conv2d(1, 16, 3))
        convs.append(torch.nn.BatchNorm2d(16))
        convs.append(torch.nn.ReLU(inplace=True))

        convs.append(torch.nn.Conv2d(16, 16, 3, stride=2))
        convs.append(torch.nn.BatchNorm3d(16))
        convs.append(torch.nn.ReLU(inplace=True))
        for idx in range(2):
            convs.append(torch.nn.Conv2d(16, 16, 3))
            convs.append(torch.nn.BatchNorm2d(16))
            convs.append(torch.nn.ReLU(inplace=True))

        self.feats = torch.nn.Sequential(*convs)
        self.regress = torch.nn.Sequential(torch.nn.Linear(128, 64),
                                           torch.nn.ReLU(inplace=True),
                                           torch.nn.Linear(64, 64),
                                           torch.nn.ReLU(inplace=True),
                                           torch.nn.Linear(64, 3))

    def forward(self, x):
        x = self.feats(x)
        x = x.view(-1, 128)
        return self.regress(x).view(-1, 2)

class FinalFineTune_RandC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        convs = list()
        convs.append(torch.nn.Conv2d(1, 16, 3, padding=1))
        convs.append(torch.nn.BatchNorm2d(16))
        convs.append(torch.nn.ReLU(inplace=True))

        convs.append(torch.nn.Conv2d(16, 16, 3, padding=1))
        convs.append(torch.nn.BatchNorm2d(16))
        convs.append(torch.nn.ReLU(inplace=True))
        convs.append(torch.nn.AvgPool2d(kernel_size=2))
        for idx in range(2):
            convs.append(torch.nn.Conv2d(16, 16, 3, padding=1))
            convs.append(torch.nn.BatchNorm2d(16))
            convs.append(torch.nn.ReLU(inplace=True))

        self.feats = torch.nn.Sequential(*convs)

        self.regress = torch.nn.Sequential(nn.Conv2d(16, 64, kernel_size=1),
                                           torch.nn.ReLU(inplace=True),
                                           nn.Conv2d(64, 64, kernel_size=1),
                                           torch.nn.ReLU(inplace=True),
                                           nn.Conv2d(64, 2, kernel_size=1))
        self.classification = torch.nn.Sequential(nn.Conv2d(16, 64, kernel_size=1),
                                           torch.nn.ReLU(inplace=True),
                                           nn.Conv2d(64, 64, kernel_size=1),
                                           torch.nn.ReLU(inplace=True),
                                           nn.Conv2d(64, 1, kernel_size=1),
                                                  nn.Sigmoid())

    def forward(self, x):
        x = self.feats(x)
        regr = self.regress(x)
        clss = self.classification(x)
        return clss, regr

class FinalFineTune_ResnetRandC(torch.nn.Module):
    def __init__(self, num_kernels=32, AvP = True, nr_firstconvlayers=4):
        super(FinalFineTune_ResnetRandC, self).__init__()
        C=num_kernels
        convs = list()
        convs.append(torch.nn.Conv2d(1, C, 3, padding=1))
        convs.append(torch.nn.BatchNorm2d(C))
        convs.append(torch.nn.ReLU(inplace=True))

        for idx in range(nr_firstconvlayers-1):
            convs.append(torch.nn.Conv2d(C,C, 3, padding=1))
            convs.append(torch.nn.BatchNorm2d(C))
            convs.append(torch.nn.ReLU(inplace=True))
        if AvP:
            convs.append(torch.nn.AvgPool2d(kernel_size=2))
        else:
            convs.append(torch.nn.Conv2d(C, C, 3, stride=2, padding=1))
        convs.append(torch.nn.Conv2d(C,C*2, 3, padding=1))
        convs.append(torch.nn.BatchNorm2d(C*2))
        convs.append(torch.nn.ReLU(inplace=True))

        for idx in range(nr_firstconvlayers//2 - 1):
            convs.append(torch.nn.Conv2d(C*2, C*2, 3, padding=1))
            convs.append(torch.nn.BatchNorm2d(C*2))
            convs.append(torch.nn.ReLU(inplace=True))

        self.feats = torch.nn.Sequential(*convs)

        self.regress = torch.nn.Sequential(nn.Conv2d(C*2, C*2, kernel_size=1),
                                           torch.nn.ReLU(inplace=True),
                                           nn.Conv2d(C*2, C*2, kernel_size=1),
                                           torch.nn.ReLU(inplace=True),
                                           nn.Conv2d(C*2, 2, kernel_size=1))
        self.classification = torch.nn.Sequential(nn.Conv2d(C*2, C*2, kernel_size=1),
                                           torch.nn.ReLU(inplace=True),
                                           nn.Conv2d(C*2, C*2, kernel_size=1),
                                           torch.nn.ReLU(inplace=True),
                                           nn.Conv2d(C*2, 1, kernel_size=1),
                                                  nn.Sigmoid())

    def forward(self, x):
        x = self.feats(x)
        regr = self.regress(x)
        clss = self.classification(x)
        return clss, regr

class MultiFineTune(torch.nn.Module):
    def __init__(self, nclasses, network_arch, num_kernels=32, AvP = True, nr_firstconvlayers=4):
        super().__init__()
        self.nclasses = nclasses
        self.network_arch = network_arch
        if network_arch == 'onlyConv':
            self.regressors = torch.nn.ModuleList([FinalFineTune_onlyConv() for idx in range(nclasses)])
        elif network_arch == 'AVPooling':
            self.regressors = torch.nn.ModuleList([FinalFineTune_AVPooling() for idx in range(nclasses)])
        elif network_arch == 'Strided':
            self.regressors = torch.nn.ModuleList([FinalFineTune_strided() for idx in range(nclasses)])
        elif network_arch == 'RandC':
            self.regressors = torch.nn.ModuleList([FinalFineTune_RandC() for idx in range(nclasses)])
        elif network_arch == 'ResnetRandC':
            self.regressors = torch.nn.ModuleList([FinalFineTune_ResnetRandC(num_kernels, AvP, nr_firstconvlayers) for idx in range(nclasses)])
        elif network_arch == 'ResnetRandCStrided':
            self.regressors = torch.nn.ModuleList(
            [FinalFineTune_ResnetRandC(num_kernels, AvP=False, nr_firstconvlayers=nr_firstconvlayers) for idx in range(nclasses)])

    def forward(self, x):
        assert (x.shape[1] == self.nclasses)

        if 'RandC' in self.network_arch:
            class_landmarks = torch.cat(
                [self.regressors[idx](x[:, idx][:, None])[0] for idx in range(self.nclasses)], axis=1)
            distance_landmarks = torch.cat(
                [self.regressors[idx](x[:, idx][:, None])[1] for idx in range(self.nclasses)], axis=1)
            distance_landmarks= torch.reshape(distance_landmarks, (distance_landmarks.size()[0],distance_landmarks.size()[1]//2,
                                                                   2, distance_landmarks.size()[2], distance_landmarks.size()[3]))

            # print(predicted_landmarks).shape
            return class_landmarks, distance_landmarks
        else:
            predicted_landmarks = torch.cat(
                [self.regressors[idx](x[:, idx][:, None]).view(-1, 1, 2) for idx in range(self.nclasses)], axis=1)
            return predicted_landmarks


if __name__ == '__main__':
    from torchsummary import summary

    sevxsev = False
    batch_norm = True

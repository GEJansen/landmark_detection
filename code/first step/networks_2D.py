import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvertToOutput(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(ConvertToOutput, self).__init__()
        self.convC = nn.Sequential(nn.Conv2d(in_feat, in_feat,
                                             kernel_size=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_feat, out_feat,
                                             kernel_size=1),
                                   nn.Sigmoid())
        self.convR = nn.Sequential(nn.Conv2d(in_feat, in_feat,
                                                 kernel_size=1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_feat, out_feat * 2,
                                                 kernel_size=1))

    def forward(self, inputs):
        class_final = self.convC(inputs)
        regr_final = self.convR(inputs)
        return class_final, regr_final

class ResnetBlock(nn.Module):
    """
    Modified resnetblock as in Identity Mappings in Deep Residual Networks (He et al. 2016): fully pre-activated
    """

    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        assert (padding_type == 'zero')

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1)]

        if norm_layer!=None:
            conv_block +=[norm_layer(dim, affine=True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.ReLU(inplace=True),
                       nn.Conv2d(dim, dim, kernel_size=3, padding=1)]

        if norm_layer!=None:
            conv_block +=[norm_layer(dim, affine=True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.ReLU(inplace=True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetBlockAVP(nn.Module):
    """
    Modified resnetblock as in Identity Mappings in Deep Residual Networks (He et al. 2016): fully pre-activated
    """

    def __init__(self, dimin, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlockAVP, self).__init__()
        self.build_conv_block(dimin, dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dimin, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        assert (padding_type == 'zero')

        conv_block += [nn.Conv2d(dimin, dim, kernel_size=3, padding=1)]

        if norm_layer!=None:
            conv_block +=[norm_layer(dim, affine=True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.ReLU(inplace=True),
                       nn.Conv2d(dim, dim, kernel_size=3, padding=1)]

        if norm_layer!=None:
            conv_block +=[norm_layer(dim, affine=True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.ReLU(inplace=True)]
        self.conv_block = nn.Sequential(*conv_block)

        self.shortcut = nn.Sequential()  # identity
        if dimin != dim:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    dimin,
                    dim,
                    kernel_size=1))
            self.shortcut.add_module('bn', nn.BatchNorm2d(dim))

    def forward(self, x):
        out = self.shortcut(x) + self.conv_block(x)
        return out


class ResnetBlock50(nn.Module):
    """
    Modified resnetblock as in Identity Mappings in Deep Residual Networks (He et al. 2016): fully pre-activated
    """

    def __init__(self, dim_in, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock50, self).__init__()
        self.build_conv_block(dim_in, dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim_in, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        assert (padding_type == 'zero')

        conv_block += [nn.Conv2d(dim_in, dim, kernel_size=1)]

        if norm_layer!=None:
            conv_block +=[norm_layer(dim, affine=True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.ReLU(inplace=True),
                       nn.Conv2d(dim, dim, kernel_size=3, padding=1)]

        if norm_layer!=None:
            conv_block +=[norm_layer(dim, affine=True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.ReLU(inplace=True)]
        conv_block += [nn.Conv2d(dim, dim*4, kernel_size=1)]

        if norm_layer != None:
            conv_block += [norm_layer(dim*4, affine=True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        self.conv_block = nn.Sequential(*conv_block)

        self.shortcut = nn.Sequential()  # identity
        if dim_in != dim*4:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    dim_in,
                    dim*4,
                    kernel_size=1))
            self.shortcut.add_module('bn', nn.BatchNorm2d(dim*4))

    def forward(self, x):
        out = self.conv_block(x) + self.shortcut(x)
        return out


class Resnet2D(nn.Module):
    def __init__(self, nclass, n_convpairs, n_downsampling, sevtsev, batch_norm):
        super(Resnet2D, self).__init__()
        self.C = 16
        n_convpairs = max(n_convpairs, 1)  # at least one residual block after each downsampling layer
        self.n_convpairs = n_convpairs

        self.n_downsampling = n_downsampling  # number of strided convolutional layers for downsampling
        self.first_kern = 3  # normally 7
        self.first_stride = 1  # normally 7
        if sevtsev:
            self.first_kern = 7
            self.first_stride = 2
        self.use_dropout = False  # dont use dropout after conv layers
        self.use_batchnorm = batch_norm
        self.nclass = nclass
        self.norm_layer = nn.BatchNorm2d
        if self.use_batchnorm==False:
            self.norm_layer = None
        self.layers()

    def layers(self):
        model = [nn.Conv2d(1, self.C, kernel_size=self.first_kern, padding=self.first_kern//2, stride=self.first_stride)]
        if self.use_batchnorm:
            model+=[self.norm_layer(self.C, affine=True)]
        model+=[nn.ReLU(inplace=True)]

        self.C_prevblock = self.C
        for i in range(self.n_downsampling): #number of strided conv layers
            mult = 2**i
            mult=min(mult, 8)
            model += [nn.Conv2d(self.C_prevblock, self.C * mult, kernel_size=3,
                                stride=2, padding=1)]
            if self.use_batchnorm:
                model+=[self.norm_layer(self.C * mult, affine=True)]
            model+=[nn.ReLU(inplace=True)]

            for n in range(self.n_convpairs):
                # A resnet-block contains: conv-layer + batchnormalization + conv-layer + batchnormalization
                model += [
                    ResnetBlock(self.C * mult, 'zero', norm_layer=self.norm_layer, use_dropout=self.use_dropout)]
            self.C_prevblock = self.C * mult
        self.model = nn.Sequential(*model)

        # make classification and regression branch
        # fully connected layers Classification
        self.FCC1 = nn.Sequential(nn.Conv2d(self.C_prevblock, self.C_prevblock, kernel_size=1), nn.ReLU(inplace=True))
        self.FCC2 = nn.Sequential(nn.Conv2d(self.C_prevblock, self.C_prevblock, kernel_size=1), nn.ReLU(inplace=True))
        self.outputC = nn.Sequential(nn.Conv2d(self.C_prevblock, self.nclass, kernel_size=1), nn.Sigmoid())

        # fully connected layers Regression
        self.FCR1 = nn.Sequential(nn.Conv2d(self.C_prevblock, self.C_prevblock, kernel_size=1), nn.ReLU(inplace=True))
        self.FCR2 = nn.Sequential(nn.Conv2d(self.C_prevblock, self.C_prevblock, kernel_size=1), nn.ReLU(inplace=True))
        self.outputR = nn.Conv2d(self.C_prevblock, (self.nclass * 2), kernel_size=1)

    def forward(self, input):
        intermediate_output = []
        out = self.model(input)

        # classification branch
        class_out = self.FCC1(out)
        class_out = self.FCC2(class_out)
        class_out = self.outputC(class_out)

        # regression branch
        regr_out = self.FCR1(out)
        regr_out = self.FCR2(regr_out)
        regr_out = self.outputR(regr_out)
        if self.nclass> 1: # multi-landmark so reshape
            regr_out = regr_out.view(regr_out.shape[0],
                                     self.nclass, 2,
                                     regr_out.shape[-2],
                                     regr_out.shape[-1])

        intermediate_output.append([class_out, regr_out])
        return intermediate_output

    def count_parameters(self, trainable):
        if trainable:
            pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            pytorch_total_params = sum(p.numel() for p in self.parameters())
        return pytorch_total_params

class Resnet2D_DeepSupervision(nn.Module):
    def __init__(self, nclass, n_convpairs, n_downsampling, intermediate_layer_nrs, prod_class, sevtsev, batch_norm):
        super(Resnet2D_DeepSupervision, self).__init__()
        self.C = 16
        n_convpairs = max(n_convpairs, 1)  # at least one residual block after each downsampling layer
        self.n_convpairs = n_convpairs

        self.n_downsampling = n_downsampling  # number of strided convolutional layers for downsampling
        self.first_kern = 3  # normally 7
        self.first_stride = 1  # normally 7
        if sevtsev:
            self.first_kern = 7
            self.first_stride = 2
        self.use_dropout = False  # dont use dropout after conv layers
        self.use_batchnorm = batch_norm
        self.nclass = nclass
        self.norm_layer = nn.BatchNorm2d
        if self.use_batchnorm==False:
            self.norm_layer = None
        self.prod_class = prod_class  # bool indicating whether classification results of intermediate layers should be
        # multiplied

        # list containing after which pooling layers a result needs to computed
        self.intermediate_layer_nrs = intermediate_layer_nrs

        # make list containing all intermediate output layers
        self.intermediate_outputlayers = []
        # make list containing after which model layers intermediate output layers should be put
        self.model_layer_numbers = []
        # make list containing the upsampling layers
        self.up_layers = []

        self.layers()

    def layers(self):
        model = [nn.Conv2d(1, self.C, kernel_size=self.first_kern, padding=self.first_kern//2, stride=self.first_stride)]
        if self.use_batchnorm:
            model+=[self.norm_layer(self.C, affine=True)]
        model+=[nn.ReLU(inplace=True)]

        self.C_prevblock = self.C
        for i in range(self.n_downsampling): #number of strided conv layers
            mult = 2**i
            mult=min(mult, 8)
            model += [nn.Conv2d(self.C_prevblock, self.C * mult, kernel_size=3,
                                stride=2, padding=1)]
            if self.use_batchnorm:
                model += [self.norm_layer(self.C * mult, affine=True)]
            model += [nn.ReLU(inplace=True)]

            for n in range(self.n_convpairs):
                # A resnet-block contains: conv-layer + batchnormalization + conv-layer + batchnormalization
                model += [
                    ResnetBlock(self.C * mult, 'zero', norm_layer=self.norm_layer, use_dropout=self.use_dropout)]

            # Does an intermediate output layer need to be added?
            if i in self.intermediate_layer_nrs:
                self.intermediate_outputlayers.append(ConvertToOutput(self.C * mult, self.nclass))
                self.model_layer_numbers.append(len(model) - 1)
            self.C_prevblock = self.C * mult

        self.intermediate_outputlayers = nn.ModuleList(self.intermediate_outputlayers)
        self.model = nn.Sequential(*model)

        # add the correct upsampling layers so the result can be upsampled and multiplied with the next (intermediate) output
        old_scale = self.n_downsampling - 1
        for scale in reversed(self.intermediate_layer_nrs):
            new_scale = old_scale - scale
            self.up_layers.append(nn.Upsample(scale_factor=2 ** new_scale, mode='bilinear'))
            old_scale = scale
        self.up_layers = nn.ModuleList(self.up_layers)

        # make classification and regression branch
        # fully connected layers Classification
        self.FCC1 = nn.Sequential(nn.Conv2d(self.C_prevblock, self.C_prevblock, kernel_size=1), nn.ReLU(inplace=True))
        self.FCC2 = nn.Sequential(nn.Conv2d(self.C_prevblock, self.C_prevblock, kernel_size=1), nn.ReLU(inplace=True))
        self.outputC = nn.Sequential(nn.Conv2d(self.C_prevblock, self.nclass, kernel_size=1), nn.Sigmoid())

        # fully connected layers Regression
        self.FCR1 = nn.Sequential(nn.Conv2d(self.C_prevblock, self.C_prevblock, kernel_size=1), nn.ReLU(inplace=True))
        self.FCR2 = nn.Sequential(nn.Conv2d(self.C_prevblock, self.C_prevblock, kernel_size=1), nn.ReLU(inplace=True))
        self.outputR = nn.Conv2d(self.C_prevblock, (self.nclass * 2), kernel_size=1)

    def forward(self, input):
        intermediate_output = []
        # list that will contain all network output
        # in pairs of [classification_output, regression_output]

        intermediate_idx = 0
        out = self.model[0](input)  # first network layer

        for layer_idx in range(1, len(self.model)):
            out = self.model[layer_idx](out)
            if layer_idx == self.model_layer_numbers[intermediate_idx]:
                # found layer after which an intermediate output layer should come
                c_intermediate, r_intermediate = self.intermediate_outputlayers[intermediate_idx](out)
                if self.nclass > 1:  # multi-landmark so reshape
                    r_intermediate = r_intermediate.view(r_intermediate.shape[0],
                                                         self.nclass, 2,
                                                         r_intermediate.shape[-2],
                                                         r_intermediate.shape[-1])
                intermediate_output.append([c_intermediate, r_intermediate])
                intermediate_idx += 1
                intermediate_idx = min(intermediate_idx, len(self.model_layer_numbers) - 1)

        # classification branch
        class_out = self.FCC1(out)
        class_out = self.FCC2(class_out)
        class_out = self.outputC(class_out)

        # regression branch
        regr_out = self.FCR1(out)
        regr_out = self.FCR2(regr_out)
        regr_out = self.outputR(regr_out)
        if self.nclass> 1: # multi-landmark so reshape
            regr_out = regr_out.view(regr_out.shape[0],
                                     self.nclass, 2,
                                     regr_out.shape[-2],
                                     regr_out.shape[-1])

        intermediate_output.append([class_out, regr_out])

        if self.prod_class:  # classification of network output should be upscaled and multiplied
            up_idx = 0
            for i in range(len(intermediate_output) - 1, 0, -1):  # go over results in reversed order
                c_out_lowres, r_out_lowres = intermediate_output[i]

                # multiply upsampled classification output with higher resolution classification output and put back in the list
                c_out_highres = self.up_layers[up_idx](c_out_lowres)
                c_combined = intermediate_output[i - 1][0] * c_out_highres
                intermediate_output[i - 1][0] = c_combined
                up_idx += 1

        return intermediate_output

    def count_parameters(self, trainable):
        if trainable:
            pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            pytorch_total_params = sum(p.numel() for p in self.parameters())
        return pytorch_total_params

class Resnet2D_34(nn.Module):
    def __init__(self, nclass, n_convpairs=None, n_downsampling=4, sevtsev=True, batch_norm=True):
        super(Resnet2D_34, self).__init__()
        self.C = 16
        self.n_convpairs = [3,4,6,3]

        self.n_downsampling = n_downsampling  # number of strided convolutional layers for downsampling
        self.first_kern = 3  # normally 7
        self.first_stride = 1  # normally 7
        if sevtsev:
            self.first_kern = 7
            self.first_stride = 2
        self.use_dropout = False  # dont use dropout after conv layers
        self.use_batchnorm = batch_norm
        self.nclass = nclass
        self.norm_layer = nn.BatchNorm2d
        if self.use_batchnorm==False:
            self.norm_layer = None
        self.layers()

    def layers(self):
        model = [nn.Conv2d(1, self.C, kernel_size=self.first_kern, padding=self.first_kern//2, stride=self.first_stride)]
        if self.use_batchnorm:
            model+=[self.norm_layer(self.C, affine=True)]
        model+=[nn.ReLU(inplace=True)]

        self.C_prevblock = self.C
        for i in range(self.n_downsampling): #number of strided conv layers
            mult = 2**(i+1)
            model += [nn.Conv2d(self.C_prevblock, self.C * mult, kernel_size=3,
                                stride=2, padding=1)]
            if self.use_batchnorm:
                model+=[self.norm_layer(self.C * mult, affine=True)]
            model+=[nn.ReLU(inplace=True)]

            for n in range(self.n_convpairs[i]):
                # A resnet-block contains: conv-layer + batchnormalization + conv-layer + batchnormalization
                model += [
                    ResnetBlock(self.C * mult, 'zero', norm_layer=self.norm_layer, use_dropout=self.use_dropout)]
            self.C_prevblock = self.C * mult
        self.model = nn.Sequential(*model)

        # make classification and regression branch
        # fully connected layers Classification
        self.FCC1 = nn.Sequential(nn.Conv2d(self.C_prevblock, self.C_prevblock, kernel_size=1), nn.ReLU(inplace=True))
        self.FCC2 = nn.Sequential(nn.Conv2d(self.C_prevblock, self.C_prevblock, kernel_size=1), nn.ReLU(inplace=True))
        self.outputC = nn.Sequential(nn.Conv2d(self.C_prevblock, self.nclass, kernel_size=1), nn.Sigmoid())

        # fully connected layers Regression
        self.FCR1 = nn.Sequential(nn.Conv2d(self.C_prevblock, self.C_prevblock, kernel_size=1), nn.ReLU(inplace=True))
        self.FCR2 = nn.Sequential(nn.Conv2d(self.C_prevblock, self.C_prevblock, kernel_size=1), nn.ReLU(inplace=True))
        self.outputR = nn.Conv2d(self.C_prevblock, (self.nclass * 2), kernel_size=1)

    def forward(self, input):
        intermediate_output = []
        out = self.model(input)

        # classification branch
        class_out = self.FCC1(out)
        class_out = self.FCC2(class_out)
        class_out = self.outputC(class_out)

        # regression branch
        regr_out = self.FCR1(out)
        regr_out = self.FCR2(regr_out)
        regr_out = self.outputR(regr_out)
        if self.nclass> 1: # multi-landmark so reshape
            regr_out = regr_out.view(regr_out.shape[0],
                                     self.nclass, 2,
                                     regr_out.shape[-2],
                                     regr_out.shape[-1])

        intermediate_output.append([class_out, regr_out])
        return intermediate_output

    def count_parameters(self, trainable):
        if trainable:
            pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            pytorch_total_params = sum(p.numel() for p in self.parameters())
        return pytorch_total_params

class Resnet2D_34_AVP(nn.Module):
    def __init__(self, nclass, n_convpairs=None, n_downsampling=4, sevtsev=True, batch_norm=True):
        super(Resnet2D_34_AVP, self).__init__()
        self.C = 16
        self.n_convpairs = [3,4,6,3]

        self.n_downsampling = n_downsampling  # number of strided convolutional layers for downsampling
        self.first_kern = 3  # normally 7
        self.first_stride = 1  # normally 7
        if sevtsev:
            self.first_kern = 7
            self.first_stride = 2
        self.use_dropout = False  # dont use dropout after conv layers
        self.use_batchnorm = batch_norm
        self.nclass = nclass
        self.norm_layer = nn.BatchNorm2d
        if self.use_batchnorm==False:
            self.norm_layer = None
        self.layers()

    def layers(self):
        model = [nn.Conv2d(1, self.C, kernel_size=self.first_kern, padding=self.first_kern//2, stride=self.first_stride)]
        if self.use_batchnorm:
            model+=[self.norm_layer(self.C, affine=True)]
        model+=[nn.ReLU(inplace=True)]

        self.C_prevblock = self.C
        for i in range(self.n_downsampling): #number of strided conv layers
            mult = 2**(i+1)
            model += [torch.nn.AvgPool2d(kernel_size=2)]
            if self.use_batchnorm:
                model+=[self.norm_layer(self.C_prevblock, affine=True)]
            model+=[nn.ReLU(inplace=True)]

            for n in range(self.n_convpairs[i]):
                # A resnet-block contains: conv-layer + batchnormalization + conv-layer + batchnormalization
                model += [
                    ResnetBlockAVP(self.C_prevblock, self.C * mult, 'zero', norm_layer=self.norm_layer, use_dropout=self.use_dropout)]
                self.C_prevblock = self.C * mult
            self.C_prevblock = self.C * mult
        self.model = nn.Sequential(*model)

        # make classification and regression branch
        # fully connected layers Classification
        self.FCC1 = nn.Sequential(nn.Conv2d(self.C_prevblock, self.C_prevblock, kernel_size=1), nn.ReLU(inplace=True))
        self.FCC2 = nn.Sequential(nn.Conv2d(self.C_prevblock, self.C_prevblock, kernel_size=1), nn.ReLU(inplace=True))
        self.outputC = nn.Sequential(nn.Conv2d(self.C_prevblock, self.nclass, kernel_size=1), nn.Sigmoid())

        # fully connected layers Regression
        self.FCR1 = nn.Sequential(nn.Conv2d(self.C_prevblock, self.C_prevblock, kernel_size=1), nn.ReLU(inplace=True))
        self.FCR2 = nn.Sequential(nn.Conv2d(self.C_prevblock, self.C_prevblock, kernel_size=1), nn.ReLU(inplace=True))
        self.outputR = nn.Conv2d(self.C_prevblock, (self.nclass * 2), kernel_size=1)

    def forward(self, input):
        intermediate_output = []
        out = self.model(input)

        # classification branch
        class_out = self.FCC1(out)
        class_out = self.FCC2(class_out)
        class_out = self.outputC(class_out)

        # regression branch
        regr_out = self.FCR1(out)
        regr_out = self.FCR2(regr_out)
        regr_out = self.outputR(regr_out)
        if self.nclass> 1: # multi-landmark so reshape
            regr_out = regr_out.view(regr_out.shape[0],
                                     self.nclass, 2,
                                     regr_out.shape[-2],
                                     regr_out.shape[-1])

        intermediate_output.append([class_out, regr_out])
        return intermediate_output

    def count_parameters(self, trainable):
        if trainable:
            pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            pytorch_total_params = sum(p.numel() for p in self.parameters())
        return pytorch_total_params

class Resnet2D_50(nn.Module):
    def __init__(self, nclass, n_convpairs=None, n_downsampling=4, sevtsev=True, batch_norm=True):
        super(Resnet2D_50, self).__init__()
        self.C = 16
        self.n_convpairs = [3,4,6,3]

        self.n_downsampling = n_downsampling  # number of strided convolutional layers for downsampling
        self.first_kern = 3  # normally 7
        self.first_stride = 1  # normally 7
        if sevtsev:
            self.first_kern = 7
            self.first_stride = 2
        self.use_dropout = False  # dont use dropout after conv layers
        self.use_batchnorm = batch_norm
        self.nclass = nclass
        self.norm_layer = nn.BatchNorm2d
        if self.use_batchnorm==False:
            self.norm_layer = None
        self.layers()

    def layers(self):
        model = [nn.Conv2d(1, self.C, kernel_size=self.first_kern, padding=self.first_kern//2, stride=self.first_stride)]
        if self.use_batchnorm:
            model+=[self.norm_layer(self.C, affine=True)]
        model+=[nn.ReLU(inplace=True)]

        self.C_prevblock = self.C
        for i in range(self.n_downsampling): #number of strided conv layers
            mult = 2**(i+1)
            model += [nn.Conv2d(self.C_prevblock, self.C * mult, kernel_size=3,
                                stride=2, padding=1)]
            if self.use_batchnorm:
                model+=[self.norm_layer(self.C * mult, affine=True)]
            model+=[nn.ReLU(inplace=True)]

            nr_feat=self.C*mult
            for n in range(self.n_convpairs[i]):
                # A resnet-block contains: conv-layer + batchnormalization + conv-layer + batchnormalization
                model += [
                    ResnetBlock50(nr_feat, self.C * mult, 'zero', norm_layer=self.norm_layer, use_dropout=self.use_dropout)]
                nr_feat= self.C * mult * 4
            self.C_prevblock = self.C * mult * 4
        self.model = nn.Sequential(*model)

        # make classification and regression branch
        # fully connected layers Classification
        self.FCC1 = nn.Sequential(nn.Conv2d(self.C_prevblock, self.C_prevblock, kernel_size=1), nn.ReLU(inplace=True))
        self.FCC2 = nn.Sequential(nn.Conv2d(self.C_prevblock, self.C_prevblock, kernel_size=1), nn.ReLU(inplace=True))
        self.outputC = nn.Sequential(nn.Conv2d(self.C_prevblock, self.nclass, kernel_size=1), nn.Sigmoid())

        # fully connected layers Regression
        self.FCR1 = nn.Sequential(nn.Conv2d(self.C_prevblock, self.C_prevblock, kernel_size=1), nn.ReLU(inplace=True))
        self.FCR2 = nn.Sequential(nn.Conv2d(self.C_prevblock, self.C_prevblock, kernel_size=1), nn.ReLU(inplace=True))
        self.outputR = nn.Conv2d(self.C_prevblock, (self.nclass * 2), kernel_size=1)

    def forward(self, input):
        intermediate_output = []
        out = self.model(input)

        # classification branch
        class_out = self.FCC1(out)
        class_out = self.FCC2(class_out)
        class_out = self.outputC(class_out)

        # regression branch
        regr_out = self.FCR1(out)
        regr_out = self.FCR2(regr_out)
        regr_out = self.outputR(regr_out)
        if self.nclass> 1: # multi-landmark so reshape
            regr_out = regr_out.view(regr_out.shape[0],
                                     self.nclass, 2,
                                     regr_out.shape[-2],
                                     regr_out.shape[-1])

        intermediate_output.append([class_out, regr_out])
        return intermediate_output

    def count_parameters(self, trainable):
        if trainable:
            pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            pytorch_total_params = sum(p.numel() for p in self.parameters())
        return pytorch_total_params

def Network(n, nclass, n_convpairs, n_downsampling, intermediate_layer_nrs=None, prod_class=None, sevtsev=False,
            batch_norm=True, AVP=False):
    rf_list = []
    conv_kernelsize=4
    stride=1
    extra_ds=0
    if sevtsev:
        conv_kernelsize=7
        stride=2
        extra_ds=1

    if n == 'resnet':
        model = Resnet2D(nclass=nclass, n_convpairs=n_convpairs, n_downsampling=n_downsampling, sevtsev=sevtsev,
                         batch_norm=batch_norm)
    elif n=='resnet_deepsupervision':
        model = Resnet2D_DeepSupervision(nclass=nclass, n_convpairs=n_convpairs, n_downsampling=n_downsampling,
                                 intermediate_layer_nrs=intermediate_layer_nrs, prod_class=prod_class, sevtsev=sevtsev,
                                         batch_norm=batch_norm)
        if intermediate_layer_nrs:
            for x in range(len(intermediate_layer_nrs)):
                rf = 1
                for i in range(intermediate_layer_nrs[x]+1):
                    rf += (n_convpairs * 2 * 2)
                    rf += 1
                    rf *= 2
                rf += (conv_kernelsize //2)
                rf *= stride
                rf_list.append((rf, 2**(intermediate_layer_nrs[x]+1+extra_ds)))
    elif n == 'resnet_34':
        conv_kernelsize = 7
        n_downsampling=4
        stride = 2
        extra_ds = 1
        if AVP:
            model = Resnet2D_34_AVP(nclass=nclass)
        else:
            model = Resnet2D_34(nclass=nclass)
        n_convpairs=[3,4,6,3]
        rf = 1
        for i in range(n_downsampling): #4 times downsampling
            rf += (n_convpairs[i] * 2 * 2)
            rf += 1
            rf *= 2
        rf += (conv_kernelsize // 2)
        rf *= stride
        rf_list.append((rf, 2 ** (n_downsampling + extra_ds)))
        return model, np.asarray(rf_list)
    elif n == 'resnet_50':
        conv_kernelsize = 7
        n_downsampling = 4
        stride = 2
        extra_ds = 1
        model = Resnet2D_50(nclass=nclass)
        n_convpairs=[3,4,6,3]
        rf = 1
        for i in range(n_downsampling): #4 times downsampling
            rf += (n_convpairs[i] * 1 * 2)
            rf += 1
            rf *= 2
        rf += (conv_kernelsize // 2)
        rf *= stride
        rf_list.append((rf, 2 ** (n_downsampling + extra_ds)))
        return model, np.asarray(rf_list)
    else:
        print("Warning: Unknown network architecture:\n%s" % n)
        sys.exit()

    rf = 1
    for i in range(n_downsampling):
        rf += (n_convpairs * 2 * 2)
        rf +=1
        rf *= 2
    rf += (conv_kernelsize //2)
    rf *= stride
    rf_list.append((rf, 2 ** (n_downsampling+extra_ds)))

    return model, np.asarray(rf_list)


if __name__ == '__main__':
    # network = Resnet2D(nclass=19, n_convpairs=2, n_downsampling=5, sevtsev=True, batch_norm=True)
    network = Resnet2D_50(nclass=19)
    # network = Resnet2D_DeepSupervision(nclass=19, n_convpairs=2, n_downsampling=5, intermediate_layer_nrs=[0,1,2,3,4],prod_class=True, sevtsev=False, batch_norm=True)
    network = network.to("cuda")
    print(network)
    params = list(network.parameters())
    print(network.count_parameters(trainable=True))
    n=32*18 #moet een meervoud van 32 zijn

    input = torch.randn(4, 1,n, n).to('cuda')
    intermediate_output = network(input)

    print(len(intermediate_output))
    print(intermediate_output[0][0].shape)
    print(intermediate_output[0][1].shape)
    print(intermediate_output[1][1].shape)
    print(intermediate_output[2][1].shape)
    print(intermediate_output[3][1].shape)
    print(intermediate_output[4][1].shape)
    print(intermediate_output[5][1].shape)



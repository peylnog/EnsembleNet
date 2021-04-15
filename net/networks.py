import torch
import torch.nn as nn
from torch.nn import *



def conv_block_size_3(in_dim,out_dim, bn = False):
    layer = nn.Sequential()
    layer.add_module( "conv1",nn.Conv2d(in_dim ,out_dim , kernel_size= 3 , padding= 1, stride=1))
    layer.add_module('relu', nn.ReLU(False))

    if bn:
        layer.add_module('bn', nn.BatchNorm2d(out_dim))

    return layer


def conv_blocks_size_3(in_dim,out_dim, Use_pool = False ,Maxpool = True,bn = True):
    layer = nn.Sequential()
    layer.add_module( "conv1",nn.Conv2d(in_dim ,out_dim , kernel_size= 3 , padding= 1, stride=1))
    if bn:
        layer.add_module('bn' ,nn.BatchNorm2d(out_dim))
    layer.add_module('relu', nn.ReLU(False))


    layer.add_module( "conv2",nn.Conv2d(out_dim ,out_dim , kernel_size= 3 , padding= 1, stride=1))
    if bn:
        layer.add_module('bn' ,nn.BatchNorm2d(out_dim))
    layer.add_module('relu', nn.ReLU(False))

    if Use_pool :
        if Maxpool:
            layer.add_module("Maxpool",nn.MaxPool2d(kernel_size=2,stride=2))
        else:
            layer.add_module("Avgpool", nn.AvgPool2d(kernel_size=2, stride=2))
    return layer



def res_net_blocks(in_dim,out_dim, Use_pool = False ,Maxpool = True,bn = True ,ker_size =3 , padding = 1 ,stride = 1):
    layer = nn.Sequential()
    layer.add_module( "conv1",nn.Conv2d(in_dim ,out_dim , kernel_size= ker_size , padding= padding, stride=stride))
    if bn:
        layer.add_module('bn' ,nn.BatchNorm2d(out_dim))
    layer.add_module('relu', nn.ReLU(False))


    layer.add_module( "conv2",nn.Conv2d(out_dim ,out_dim , kernel_size= ker_size , padding= padding, stride=stride))
    if bn:
        layer.add_module('bn' ,nn.BatchNorm2d(out_dim))
    layer.add_module('relu', nn.ReLU(False))

    return layer


def conv_blocks_size_5(in_dim,out_dim, Use_pool = False ,Maxpool = True ,bn = True):
    layer = nn.Sequential()
    layer.add_module( "conv1",nn.Conv2d(in_dim ,out_dim , kernel_size= 5, padding= 2, stride=1))

    if bn:
        layer.add_module('bn' ,nn.BatchNorm2d(out_dim))

    layer.add_module('relu', nn.ReLU(False))



    layer.add_module( "conv2",nn.Conv2d(out_dim ,out_dim , kernel_size= 5, padding= 2, stride=1))
    # layer.add_module('relu', nn.ReLU(False))

    if bn:
        layer.add_module('bn' ,nn.BatchNorm2d(out_dim))
    layer.add_module('relu', nn.ReLU(False))

    if Use_pool :
        if Maxpool:
            layer.add_module("Maxpool",nn.MaxPool2d(kernel_size=2,stride=2))
        else:
            layer.add_module("Avgpool", nn.AvgPool2d(kernel_size=2, stride=2))
    return layer




def conv_blocks_size_7(in_dim,out_dim, Use_pool = False ,Maxpool = True , bn = True):
    layer = nn.Sequential()
    layer.add_module( "conv1",nn.Conv2d(in_dim ,out_dim , kernel_size= 7 , padding= 3, stride=1))
    if bn:
        layer.add_module('bn', nn.BatchNorm2d(out_dim))
    layer.add_module('relu', nn.ReLU(False))


    layer.add_module( "conv2",nn.Conv2d(out_dim ,out_dim , kernel_size= 7 , padding= 3, stride=1))
    if bn:
        layer.add_module('bn', nn.BatchNorm2d(out_dim))
    layer.add_module('relu', nn.ReLU(False))

    if Use_pool :
        if Maxpool:
            layer.add_module("Maxpool",nn.MaxPool2d(kernel_size=2,stride=2))
        else:
            layer.add_module("Avgpool", nn.AvgPool2d(kernel_size=2, stride=2))
    return layer




def deconv_blocks_size_3(in_dim,out_dim,Use_pool=True,bn=True):
    layer = nn.Sequential()
    layer.add_module( "conv1",nn.ConvTranspose2d(in_dim , out_dim , 3 , 1, 1))
    if bn:
        layer.add_module('bn', nn.BatchNorm2d(out_dim))
    layer.add_module('relu', nn.ReLU(False))


    layer.add_module("conv2", nn.ConvTranspose2d(out_dim, out_dim ,3, 1, 1))
    if bn:
        layer.add_module('bn', nn.BatchNorm2d(out_dim))
    layer.add_module('relu', nn.ReLU(False))

    if Use_pool:
        layer.add_module("Upsamp",nn.UpsamplingNearest2d(scale_factor= 2))
    return layer




def Nonlinear_layer(in_c=0  , name="nonlinear",   bn=False ,  relu=True, LeakReLU = False , dropout=False ):
    layer = nn.Sequential()
    if relu:
        layer.add_module('%s_relu' % name, nn.ReLU(inplace=False))
    if LeakReLU:
        layer.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if bn:
        layer.add_module('%s_bn' % name, nn.BatchNorm2d(in_c))

    if dropout:
        layer.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=False))
    return layer

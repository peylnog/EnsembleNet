# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import re
import torch
import argparse
import urllib.request
from Utils.utils import *
from Utils.Vidsom import *
from Utils.model_init import *
from Utils.ssim_map import SSIM_MAP
from Utils.torch_ssim import SSIM
from torch import nn, optim
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from MyDataset.Datasets import derain_test_datasets, derain_train_datasets
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, CenterCrop, RandomCrop

from net.model_ori import w_net

parser = argparse.ArgumentParser(description="PyTorch Derain W")
# root

# root
parser.add_argument("--train", default="/scratch1/hxw170830/derain_ablation_V1/DID-MDN-datasets/train", type=str,
                    help="path to load train datasets(default: none)")
parser.add_argument("--test", default="/scratch1/hxw170830/derain_ablation_V1/DID-MDN-datasets/test", type=str,
                    help="path to load test datasets(default: none)")

parser.add_argument("--save_image_root", default='./result', type=str,
                    help="save test image root")
parser.add_argument("--save_root", default="./checkpoints", type=str,
                    help="path to save networks")
parser.add_argument("--pretrain_root", default="/scratch1/hxw170830/derain_ablation_V1/checkpoints2", type=str,
                    help="path to pretrained net1 net2 net3 root")

# hypeparameters
parser.add_argument("--batchSize", type=int, default=32, help="training batch size")
parser.add_argument("--nEpoch", type=int, default=10000, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--lr1", type=float, default=1e-4, help="Learning Rate For w")

parser.add_argument("--train_print_fre", type=int, default=50, help="frequency of print train loss on train phase")
parser.add_argument("--test_frequency", type=int, default=1, help="frequency of test")
parser.add_argument("--test_print_fre", type=int, default=400, help="frequency of print train loss on test phase")
parser.add_argument("--cuda", type=str, default="Ture", help="Use cuda?")
parser.add_argument("--gpus", type=int, default=1, help="nums of gpu to use")
parser.add_argument("--startweights", default=0, type=int, help="start number of net's weight , 0 is None")
parser.add_argument("--initmethod", default='xavier', type=str,
                    help="xavier , kaiming , normal ,orthogonal ,default : xavier")
parser.add_argument("--startepoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--works", type=int, default=8, help="Number of works for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum, Default: 0.9")
parser.add_argument("--report", default=False, type=bool, help="report to wechat")
parser.add_argument("--save_image", default=False, type=bool, help="save test image")
parser.add_argument("--pretrain_epoch", default=[93, 169, 123], type=list, help="pretrained epoch for Net1 Net2 Net3")


def main():
    global opt, Net1, Net2, Net3, Net4, RefineNet, criterion_mse, criterion_ssim_map, criterion_ssim, criterion_ace
    global w,criterion_l1
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    cudnn.benchmark = True

    print("==========> Loading datasets")

    train_dataset = derain_train_datasets(data_root=opt.train, transform=Compose([
        ToTensor()
    ]))

    test_dataset = derain_test_datasets(opt.test, transform=Compose([
        ToTensor()
    ]))

    training_data_loader = DataLoader(dataset=train_dataset, num_workers=opt.works, batch_size=opt.batchSize,
                                      pin_memory=True, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_dataset, num_workers=opt.works, batch_size=1, pin_memory=True,
                                     shuffle=True)

    if opt.initmethod == 'orthogonal':
        init_function = weights_init_orthogonal

    elif opt.initmethod == 'kaiming':
        init_function = weights_init_kaiming

    elif opt.initmethod == 'normal':
        init_function = weights_init_normal

    else:
        init_function = weights_init_xavier

    w = w_net()
    w.apply(init_function)

    criterion_mse = nn.MSELoss()
    criterion_ssim_map = SSIM_MAP()
    criterion_ssim = SSIM()
    criterion_ace = nn.SmoothL1Loss()

    criterion_l1 = nn.L1Loss()

    print("==========> Setting GPU")
    # if cuda:
    if opt.cuda:
        w = nn.DataParallel(w, device_ids=[i for i in range(opt.gpus)]).cuda()

        criterion_ssim = criterion_ssim.cuda()
        criterion_ssim_map = criterion_ssim_map.cuda()
        criterion_mse = criterion_mse.cuda()
        criterion_ace = criterion_ace.cuda()
        criterion_l1 = criterion_l1.cuda()
    else:
        raise Exception("it takes a long time without cuda ")
    # print(net)

    # if opt.pretrain_root:
    #     if os.path.exists(opt.pretrain_root):
    #         print("=> loading net from '{}'".format(opt.pretrain_root))
    #         weights = torch.load(opt.pretrain_root +"/w/%s.pth"%opt.pretrain_epoch[0])
    #         Net1.load_state_dict(weights['state_dict'] )
    #
    #         weights = torch.load(opt.pretrain_root + "/u/%s.pth" % opt.pretrain_epoch[1])
    #         Net2.load_state_dict(weights['state_dict'] )
    #
    #         weights = torch.load(opt.pretrain_root + "/res/%s.pth" % opt.pretrain_epoch[2])
    #         Net3.load_state_dict(weights['state_dict'])
    #
    #         del weights
    #     else:
    #         print("=> no net found at '{}'".format(opt.pretrain_root))

    # weights start from early
    # if opt.startweights:
    #     if os.path.exists(opt.save_root):
    #         print("=> loading checkpoint '{}'".format(opt.save_root))
    #         weights = torch.load(opt.save_root + '/Net1/%s.pth'%opt.startweights)
    #         Net1.load_state_dict(weights["state_dict"] )
    #
    #         weights = torch.load(opt.save_root + '/Net2/%s.pth' % opt.startweights)
    #         Net2.load_state_dict(weights["state_dict"])
    #
    #         weights = torch.load(opt.save_root + '/Net3/%s.pth' % opt.startweights)
    #         Net3.load_state_dict(weights["state_dict"])
    #
    #         weights = torch.load(opt.save_root + '/Net4/%s.pth' % opt.startweights)
    #         Net4.load_state_dict(weights["state_dict"])
    #
    #         weights = torch.load(opt.save_root + '/refine/%s.pth' % opt.startweights)
    #         RefineNet.load_state_dict(weights["state_dict"])
    #
    #         del weights
    #     else:
    #         raise Exception("'{}' is not a file , Check out it again".format(opt.save_root))

    print("==========> Setting Optimizer")
    optimizerw = optim.Adam(filter(lambda p: p.requires_grad, w.parameters()), lr=opt.lr1)

    optimizer = [1, optimizerw]
    print("==========> Training")
    for epoch in range(opt.startepoch, opt.nEpoch + 1):

        if epoch > 50:
            opt.lr1 = 1e-4
            optimizer[1] = optim.Adam(filter(lambda p: p.requires_grad, w.parameters()), lr=opt.lr1)

        train(training_data_loader, optimizer, epoch)

        if epoch % opt.test_frequency == 0:
            test(testing_data_loader, epoch)


def train(training_data_loader, optimizer, epoch):
    print("training ==========> epoch =", epoch, "lr =", opt.lr1)
    w.train()
    # Net1.train()
    # Net2.train()
    # Net3.train()
    # Net4.train()
    # RefineNet.train()
    t_loss = []  # save trainloss

    for step, (data, label) in enumerate(training_data_loader, 1):
        if opt.cuda and torch.cuda.is_available():
            data = data.clone().detach().requires_grad_(True).cuda()
            label = label.cuda()
        else:
            raise Exception("it takes a long time without cuda ")
            data = data.cpu()
            label = label.cpu()

        w_out = w(data)
        ssim_loss  = 1 - criterion_ssim_map(w_out , label).mean()

        w.zero_grad()
        optimizer[1].zero_grad()

        ssim_loss.backward()
        optimizer[1].step()

        if step % opt.train_print_fre == 0:
            print("epoch{} step {} loss {:6f} ".format(epoch, step, ssim_loss.item(), ))
            t_loss.append(l1_loss.item())

    # else:
    # displaying to train loss
    # updata_epoch_loss_display( train_loss= t_loss , v_epoch= epoch , envr= "derain train")


def test(test_data_loader, epoch):
    print("------> testing")
    w.eval()

    torch.cuda.empty_cache()

    with torch.no_grad():

        test_Psnr_sum = 0.0
        test_Ssim_sum = 0.0

        # showing list
        test_Psnr_loss = []
        test_Ssim_loss = []
        dict_psnr_ssim = {}
        for test_step, (data, label, data_path) in enumerate(test_data_loader, 1):
            data = data.cuda()
            label = label.cuda()

            w_out = w(data)

            ssim_loss = 1 - criterion_ssim(w_out, label)
            Psnr, Ssim = get_psnr_ssim(w_out, label)

            Psnr = round(Psnr.item(), 4)
            Ssim = round(Ssim.item(), 4)

            test_Psnr_sum += Psnr
            test_Ssim_sum += Ssim

            # if opt.save_image == True:
            #    dict_psnr_ssim["Psnr%s_Ssim%s" % (Psnr, Ssim)] = data_path
            #    out = refineNet_out.cpu().data[0]
            #    out = ToPILImage()(out)
            #    image_number = re.findall(r'\d+', data_path[0])[1]
            #    out.save( opt.save_image_root + "/%s_pï¼š%s_sï¼š%s.jpg" % (image_number, Psnr, Ssim))
            if test_step % opt.test_print_fre == 0:
                print("epoch={}  Psnr={}  Ssim={} mseloss{}".format(epoch, Psnr, Ssim, ssim_loss.item()))
                test_Psnr_loss.append(test_Psnr_sum / test_step)
                test_Ssim_loss.append(test_Ssim_sum / test_step)

        else:
            del ssim_loss
            print("epoch={}  avr_Psnr ={}  avr_Ssim={}".format(epoch, test_Psnr_sum / test_step,
                                                               test_Ssim_sum / test_step))
            write_test_perform("./perform_test_l1.txt", test_Psnr_sum / test_step, test_Ssim_sum / test_step)
            # visdom showing
            print("---->testing over show in visdom")
            # display_Psnr_Ssim(Psnr=test_Psnr_sum / test_step, Ssim=test_Ssim_sum / test_step, v_epoch=epoch,
            # env="derain_test")

    print("epoch {} train over-----> save net".format(epoch))
    print("saving checkpoint      save_root{}".format(opt.save_root))
    if os.path.exists(opt.save_root):
        save_checkpoint(root=opt.save_root, model=w, epoch=epoch, model_stage="wl1")
        print("finish save epoch{} checkporint".format({epoch}))
    else:
        raise Exception("saveroot :{} not found , Checkout it".format(opt.save_root))
    #

    print("all epoch is over ------ ")
    print("show epoch and epoch_loss in visdom")


if __name__ == "__main__":
    os.system('clear')
    main()


# coding=utf-8
import os
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
from MyDataset.Datasets import derain_test_datasets , derain_train_datasets
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, CenterCrop, RandomCrop

from net.model import w_net as net1
from net.model import u_net as net2
from net.model import res_net as net3
from net.model import Net4 as net4
from net.model import refineNet as refine


parser = argparse.ArgumentParser(description="PyTorch Derain")
#root
parser.add_argument("--train", default="/home/ws/Desktop/PL/Derain_Dataset2018/train", type=str,
                    help="path to load train datasets(default: none)")
parser.add_argument("--test", default="/home/ws/Desktop/PL/Derain_Dataset2018/test", type=str,
                    help="path to load test datasets(default: none)")

parser.add_argument("--save_image_root", default='./result', type=str,
                    help="save test image root")
parser.add_argument("--save_root", default="/home/ws/Desktop/derain2020/checkpoints", type=str,
                    help="path to save networks")
parser.add_argument("--pretrain_root", default="/home/ws/Desktop/derain2020/checkpoints", type=str,
                    help="path to pretrained net1 net2 net3 root")

#hypeparameters
parser.add_argument("--batchSize", type=int, default=8, help="training batch size")
parser.add_argument("--nEpoch", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--lr1", type=float, default=5e-5, help="Learning Rate For pretrained net. Default=1e-5")
parser.add_argument("--p", default=0.8, type=float, help="probability of normal conditions")

parser.add_argument("--train_print_fre", type=int, default=200, help="frequency of print train loss on train phase")
parser.add_argument("--test_frequency", type=int, default=1, help="frequency of test")
parser.add_argument("--test_print_fre", type=int, default=200, help="frequency of print train loss on test phase")
parser.add_argument("--cuda",type=str, default="Ture", help="Use cuda?")
parser.add_argument("--gpus", type=int, default=1, help="nums of gpu to use")
parser.add_argument("--startweights", default= 32, type=int, help="start number of net's weight , 0 is None")
parser.add_argument("--initmethod", default='xavier', type=str, help="xavier , kaiming , normal ,orthogonal ,default : xavier")
parser.add_argument("--startepoch", default=38, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--works", type=int, default=8, help="Number of works for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum, Default: 0.9")
parser.add_argument("--report", default=False, type=bool, help="report to wechat")
parser.add_argument("--save_image", default=False, type=bool, help="save test image")
parser.add_argument("--pretrain_epoch", default=[93,169,123], type=list, help="pretrained epoch for Net1 Net2 Net3")


def main():
    global opt, Net1 , Net2 , Net3 , Net4 , RefineNet , criterion_mse , criterion_ssim_map,criterion_ssim,criterion_ace
    opt = parser.parse_args()
    print(opt)


    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    seed = 1334
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    cudnn.benchmark = True

    print("==========> Loading datasets")

    train_dataset = derain_train_datasets( data_root= opt.train, transform=Compose([
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

    Net1 = net1()
    Net1.apply(init_function)
    Net2 = net2()
    Net2.apply(init_function)
    Net3 = net3()
    Net3.apply(init_function)
    Net4 = net4()
    Net4.apply(init_function)
    RefineNet = refine()
    RefineNet.apply(init_function)



    criterion_mse = nn.MSELoss(size_average=True)
    criterion_ssim_map  = SSIM_MAP()
    criterion_ssim  = SSIM()
    criterion_ace = nn.SmoothL1Loss()

    print("==========> Setting GPU")
    #if cuda:
    if opt.cuda:
        Net1 = nn.DataParallel(Net1, device_ids=[i for i in range(opt.gpus)]).cuda()
        Net2 = nn.DataParallel(Net2, device_ids=[i for i in range(opt.gpus)]).cuda()
        Net3 = nn.DataParallel(Net3, device_ids=[i for i in range(opt.gpus)]).cuda()
        Net4 = nn.DataParallel(Net4, device_ids=[i for i in range(opt.gpus)]).cuda()
        RefineNet = nn.DataParallel(RefineNet, device_ids=[i for i in range(opt.gpus)]).cuda()

        criterion_ssim = criterion_ssim.cuda()
        criterion_ssim_map = criterion_ssim_map.cuda()
        criterion_mse= criterion_mse.cuda()
        criterion_ace = criterion_ace.cuda()
    else:
        raise Exception("it takes a long time without cuda ")
    #print(net)

    if opt.pretrain_root:
        if os.path.exists(opt.pretrain_root):
            print("=> loading net from '{}'".format(opt.pretrain_root))
            weights = torch.load(opt.pretrain_root +"/w/%s.pth"%opt.pretrain_epoch[0])
            Net1.load_state_dict(weights['state_dict'] )

            weights = torch.load(opt.pretrain_root + "/u/%s.pth" % opt.pretrain_epoch[1])
            Net2.load_state_dict(weights['state_dict'] )

            weights = torch.load(opt.pretrain_root + "/res/%s.pth" % opt.pretrain_epoch[2])
            Net3.load_state_dict(weights['state_dict'])

            del weights
        else:
            print("=> no net found at '{}'".format(opt.pretrain_root))

    # weights start from early
    if opt.startweights:
        if os.path.exists(opt.save_root):
            print("=> loading checkpoint '{}'".format(opt.save_root))
            weights = torch.load(opt.save_root + '/Net1/%s.pth'%opt.startweights)
            Net1.load_state_dict(weights["state_dict"] )

            weights = torch.load(opt.save_root + '/Net2/%s.pth' % opt.startweights)
            Net2.load_state_dict(weights["state_dict"])

            weights = torch.load(opt.save_root + '/Net3/%s.pth' % opt.startweights)
            Net3.load_state_dict(weights["state_dict"])

            weights = torch.load(opt.save_root + '/Net4/%s.pth' % opt.startweights)
            Net4.load_state_dict(weights["state_dict"])

            weights = torch.load(opt.save_root + '/refine/%s.pth' % opt.startweights)
            RefineNet.load_state_dict(weights["state_dict"])

            del weights
        else:
            raise Exception("'{}' is not a file , Check out it again".format(opt.save_root))



    print("==========> Setting Optimizer")
    optimizer1 = optim.Adam(filter(lambda p: p.requires_grad, Net1.parameters()), lr=opt.lr1)
    optimizer2 = optim.Adam(filter(lambda p: p.requires_grad, Net2.parameters()), lr=opt.lr1)
    optimizer3 = optim.Adam(filter(lambda p: p.requires_grad, Net3.parameters()), lr=opt.lr1)
    optimizer4 = optim.Adam(filter(lambda p: p.requires_grad, Net4.parameters()), lr=opt.lr)
    optimizer_Refine = optim.Adam(filter(lambda p: p.requires_grad, RefineNet.parameters()), lr=opt.lr)

    optimizer = [ 1 , optimizer1 , optimizer2 , optimizer3 , optimizer4 , optimizer_Refine ]
    print("==========> Training")
    for epoch in range(opt.startepoch, opt.nEpoch + 1):

        if epoch > 10 :
            opt.lr = 1e-4
            optimizer[1] = optim.Adam(filter(lambda p: p.requires_grad, Net1.parameters()), lr=opt.lr1)
            optimizer[2] = optim.Adam(filter(lambda p: p.requires_grad, Net2.parameters()), lr=opt.lr1)
            optimizer[3] = optim.Adam(filter(lambda p: p.requires_grad, Net3.parameters()), lr=opt.lr1)
            optimizer[4] = optim.Adam(filter(lambda p: p.requires_grad, Net4.parameters()), lr=opt.lr)
            optimizer[5] = optim.Adam(filter(lambda p: p.requires_grad, RefineNet.parameters()), lr=opt.lr)

        # train(training_data_loader, optimizer, epoch)

        if epoch % opt.test_frequency == 0 :
            test(testing_data_loader ,epoch)



def train(training_data_loader, optimizer, epoch):
    print("training ==========> epoch =", epoch, "lr =", opt.lr)
    Net1.train()
    Net2.train()
    Net3.train()
    Net4.train()
    RefineNet.train()
    t_loss = []  # save trainloss

    for step, (data, label) in enumerate(training_data_loader, 1):
        if opt.cuda and torch.cuda.is_available():
            data = data.clone().detach().requires_grad_(True).cuda()
            label = label.cuda()
        else:
            raise Exception("it takes a long time without cuda ")
            data = data.cpu()
            label = label.cpu()

        Net1_out = Net1(data)
        Net2_out = Net2(Net1_out)
        Net3_out = Net3(Net2_out)
        Net4_out = Net4( data - Net1_out  ,data - Net2_out ,data - Net3_out )
        RefineNet_out =RefineNet( Net1_out , Net2_out , Net3_out , data - Net4_out )

        init_map = torch.ones(size=Net1_out.size()).cuda()
        ssim_map1 = torch.mul(criterion_ssim_map(Net1_out , label) , init_map )
        ssim_map2 = torch.mul(criterion_ssim_map(Net2_out , label) , ssim_map1 )
        ssim_map3 = torch.mul(criterion_ssim_map(Net3_out , label) , ssim_map2 )

        loss1 = torch.mul((1 - ssim_map1) , torch.abs(Net1_out - label)).mean()
        loss2 = torch.mul((1 - ssim_map2) , torch.abs(Net2_out - label)).mean()
        loss3 = torch.mul((1 - ssim_map3) , torch.abs(Net3_out - label)).mean()

        new_loss  = torch.mul((1-criterion_ssim_map(RefineNet_out , label)) ,torch.abs(RefineNet_out-label)).mean().cuda()
        ssim_loss  = 1- criterion_ssim(RefineNet_out , label)

        loss = new_loss + 0.01 * (loss1 + loss2 +loss3) 


        del Net1_out , Net2_out , Net3_out , Net4_out
        Net1.zero_grad()
        Net2.zero_grad()
        Net3.zero_grad()
        Net4.zero_grad()
        RefineNet.zero_grad()


        optimizer[1].zero_grad()
        optimizer[2].zero_grad()
        optimizer[3].zero_grad()
        optimizer[4].zero_grad()
        optimizer[5].zero_grad()

        loss.backward()
        optimizer[1].step()
        optimizer[2].step()
        optimizer[3].step()
        optimizer[4].step()
        optimizer[5].step()


        if step % opt.train_print_fre == 0:
            print("epoch{} step {} loss {:6f}  new_loss {:6f} ssimloss {:6f}  loss1 {:6f}  loss2 {:6f}  loss3 {:6f}".format(epoch, step,
                                                                                               loss.item(),
                                                                                               new_loss.item(),
                                                                                               ssim_loss.item(),
                                                                                               loss1.item(),
                                                                                               loss2.item(),
                                                                                               loss3.item()))
            t_loss.append(loss.item())
        del loss1, loss2, loss3 , loss

    else:
        # displaying to train loss
        updata_epoch_loss_display( train_loss= t_loss , v_epoch= epoch , envr= "derain train")

import time
def test(test_data_loader, epoch):
    print("------> testing")
    Net1.eval()
    Net2.eval()
    Net3.eval()
    Net4.eval()
    RefineNet.eval()
    torch.cuda.empty_cache()
    starttime = 0
    endtime = 0
    with torch.no_grad():

        test_Psnr_sum = 0.0
        test_Ssim_sum = 0.0

        # showing list
        test_Psnr_loss = []
        test_Ssim_loss = []
        dict_psnr_ssim = {}
        starttime = time.time()
        for test_step, (data, label, data_path) in enumerate(test_data_loader, 1):
            data = data.cuda()
            label = label.cuda()

            Net1_out = Net1(data).cuda()
            Net2_out = Net2(Net1_out).cuda()
            Net3_out = Net3(Net2_out).cuda()
            Net4_out = Net4(data - Net1_out , data - Net2_out , data - Net3_out).cuda() #best rain streaks
            refineNet_out = RefineNet(Net1_out , Net2_out , Net3_out , data - Net4_out ).cuda()

            del Net1_out, Net2_out, Net3_out

            loss = criterion_mse(refineNet_out, label)
            Psnr, Ssim = get_psnr_ssim(refineNet_out, label)

            Psnr = round(Psnr.item(), 4)
            Ssim = round(Ssim.item(), 4)

            # del derain , label
            test_Psnr_sum += Psnr
            test_Ssim_sum += Ssim

            #if opt.save_image == True:
            #    dict_psnr_ssim["Psnr%s_Ssim%s" % (Psnr, Ssim)] = data_path
            #    out = refineNet_out.cpu().data[0]
            #    out = ToPILImage()(out)
            #    image_number = re.findall(r'\d+', data_path[0])[1]
            #    out.save( opt.save_image_root + "/%s_p：%s_s：%s.jpg" % (image_number, Psnr, Ssim))
            if test_step % opt.test_print_fre == 0:
                print("epoch={}  Psnr={}  Ssim={} loss{}".format(epoch, Psnr, Ssim, loss.item()))
                test_Psnr_loss.append(test_Psnr_sum / test_step)
                test_Ssim_loss.append(test_Ssim_sum / test_step)

        else:
            del loss
            print("epoch={}  avr_Psnr ={}  avr_Ssim={}".format(epoch, test_Psnr_sum / test_step,
                                                               test_Ssim_sum / test_step))
            write_test_perform("./perform_test.txt", test_Psnr_sum / test_step, test_Ssim_sum / test_step)
            # visdom showing
            print("---->testing over show in visdom")
            display_Psnr_Ssim(Psnr=test_Psnr_sum / test_step, Ssim=test_Ssim_sum / test_step, v_epoch=epoch,
                              env="derain_test")
        endtime = time.time()

    print("----------TestTime:{}".format(endtime - starttime))
    print("epoch {} train over-----> save net".format(epoch))
    print("saving checkpoint      save_root{}".format(opt.save_root))
    if os.path.exists(opt.save_root):
        save_checkpoint(root=opt.save_root, model=Net1, epoch=epoch, model_stage="Net1")
        save_checkpoint(root=opt.save_root, model=Net2, epoch=epoch, model_stage="Net2")
        save_checkpoint(root=opt.save_root, model=Net3, epoch=epoch, model_stage="Net3")
        save_checkpoint(root=opt.save_root, model=Net4, epoch=epoch, model_stage="Net4")
        save_checkpoint(root=opt.save_root, model=RefineNet, epoch=epoch, model_stage="refine")

        print("finish save epoch{} checkporint".format({epoch}))
    else:
        raise  Exception("saveroot :{} not found , Checkout it".format(opt.save_root))
    #

    print("all epoch is over ------ ")
    print("show epoch and epoch_loss in visdom")

if __name__ == "__main__":
    os.system('clear')
    main()

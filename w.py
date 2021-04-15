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

from net.model import w_net



parser = argparse.ArgumentParser(description="PyTorch Derain W")
#root
parser.add_argument("--train", default="/home/ws/Desktop/PL/Derain_Dataset2018/train", type=str,
                    help="path to load train datasets(default: none)")
parser.add_argument("--test", default="/home/ws/Desktop/PL/Derain_Dataset2018/test", type=str,
                    help="path to load test datasets(default: none)")

parser.add_argument("--save_image_root", default='./result_mseloss', type=str,
                    help="save test image root")
parser.add_argument("--save_root", default="/home/ws/Desktop/derain2020/checkpoints", type=str,
                    help="path to save networks")
parser.add_argument("--pretrain_root", default="/home/ws/Desktop/derain2020/checkpoints", type=str,
                    help="path to pretrained net1 net2 net3 root")

#hypeparameters
parser.add_argument("--batchSize", type=int, default=8, help="training batch size")
parser.add_argument("--nEpoch", type=int, default=10000, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--lr1", type=float, default=1e-4, help="Learning Rate For w")

parser.add_argument("--train_print_fre", type=int, default=500, help="frequency of print train loss on train phase")
parser.add_argument("--test_frequency", type=int, default=1, help="frequency of test")
parser.add_argument("--test_print_fre", type=int, default=200, help="frequency of print train loss on test phase")
parser.add_argument("--cuda",type=str, default="Ture", help="Use cuda?")
parser.add_argument("--gpus", type=int, default=1, help="nums of gpu to use")
parser.add_argument("--startweights", default=0, type=int, help="start number of net's weight , 0 is None")
parser.add_argument("--initmethod", default='xavier', type=str, help="xavier , kaiming , normal ,orthogonal ,default : xavier")
parser.add_argument("--startepoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--works", type=int, default=4, help="Number of works for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum, Default: 0.9")
parser.add_argument("--report", default=False, type=bool, help="report to wechat")
parser.add_argument("--save_image", default=False, type=bool, help="save test image")
parser.add_argument("--pretrain_epoch", default=[93,169,123], type=list, help="pretrained epoch for Net1 Net2 Net3")

from torchvision import models
from PIL import Image



class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        #for param in self.parameters():
        #    param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out


class PerceptualLoss:
    def __init__(self , content_layer = 1 , content_layers = 0):
        self.content_layer = content_layer
        self.content_layers = content_layers

        self.vgg = nn.DataParallel(Vgg16()).cuda()
        self.vgg.eval()
        self.L1Loss = nn.DataParallel(nn.L1Loss()).cuda()
        self.L1Loss_sum = nn.DataParallel(nn.L1Loss(reduction='sum')).cuda()

    def __call__(self, x, y_hat):
        # b, c, h, w = x.shape
        y_content_features = self.vgg(x)
        y_hat_features = self.vgg(y_hat)

        recon = y_content_features[self.content_layer].cuda()
        recon_hat = y_hat_features[self.content_layer].cuda()

        recon1 = y_content_features[self.content_layers].cuda()
        recon_hat1 = y_hat_features[self.content_layers].cuda()

        L_content = self.L1Loss_sum(recon_hat, recon).cuda()
        L_content1 = self.L1Loss_sum(recon_hat1, recon1).cuda()


        return L_content+L_content1

def main():
    global opt, Net1 , Net2 , Net3 , Net4 , RefineNet , criterion_mse , criterion_ssim_map,criterion_ssim,criterion_ace
    global w,criterion_p
    opt = parser.parse_args()
    print(opt)


    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")


    cudnn.benchmark = True

    print("==========> Loading datasets")

    train_dataset = derain_train_datasets( data_root= opt.train, transform=Compose([
        ToTensor()
    ]))

    test_dataset = derain_test_datasets(opt.test, transform=Compose([
        ToTensor()
    ]))

    training_data_loader = DataLoader(dataset=train_dataset, num_workers=opt.works, batch_size=opt.batchSize,
                                      pin_memory=False, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_dataset, num_workers=opt.works, batch_size=1, pin_memory=False,
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
    criterion_ssim_map  = SSIM_MAP()
    criterion_ssim  = SSIM()
    criterion_ace = nn.L1Loss()
    criterion_p = PerceptualLoss()
    print("==========> Setting GPU")
    #if cuda:
    if opt.cuda:
        w = nn.DataParallel(w, device_ids=[i for i in range(opt.gpus)]).cuda()


        criterion_ssim = criterion_ssim.cuda()
        criterion_ssim_map = criterion_ssim_map.cuda()
        criterion_mse= criterion_mse.cuda()
        criterion_ace = criterion_ace.cuda()
        #criterion_p = criterion_p.cuda()
    else:
        raise Exception("it takes a long time without cuda ")
    #print(net)

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
    if opt.startweights:
        if os.path.exists(opt.save_root):
            print("=> loading checkpoint '{}'".format(opt.save_root))
            weights = torch.load(opt.save_root + '/%s.pth'%opt.startweights)
            w.load_state_dict(weights["state_dict"] )
    
            # weights = torch.load(opt.save_root + '/Net2/%s.pth' % opt.startweights)
            # Net2.load_state_dict(weights["state_dict"])
    
            # weights = torch.load(opt.save_root + '/Net3/%s.pth' % opt.startweights)
            # Net3.load_state_dict(weights["state_dict"])
    
            # weights = torch.load(opt.save_root + '/Net4/%s.pth' % opt.startweights)
            # Net4.load_state_dict(weights["state_dict"])
    
            # weights = torch.load(opt.save_root + '/refine/%s.pth' % opt.startweights)
            # RefineNet.load_state_dict(weights["state_dict"])
    
            del weights
        else:
            raise Exception("'{}' is not a file , Check out it again".format(opt.save_root))



    print("==========> Setting Optimizer")
    optimizerw = optim.Adam(filter(lambda p: p.requires_grad, w.parameters()), lr=opt.lr1)


    optimizer = [ 1 , optimizerw ]
    print("==========> Training")
    for epoch in range(opt.startepoch, opt.nEpoch + 1):

        if epoch > 400 :
            opt.lr1 = 1e-4
            optimizer[1] = optim.Adam(filter(lambda p: p.requires_grad, w.parameters()), lr=opt.lr1)


        train(training_data_loader, optimizer, epoch)

        if epoch % opt.test_frequency == 0 :
            test(testing_data_loader ,epoch)



def train(training_data_loader, optimizer, epoch):
    print("training ==========> epoch =", epoch, "lr1 =", opt.lr1)
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
        loss = criterion_p(w_out , label)
        #new_loss  = torch.mul((1-criterion_ssim_map(w_out , label)) , torch.abs(w_out-label)).mean().cuda()

        #loss = new_loss
        # del Net1_out , Net2_out , Net3_out , Net4_out

        w.zero_grad()
        optimizer[1].zero_grad()
        loss.backward()
        optimizer[1].step()



        if step % opt.train_print_fre == 0:
            print("epoch{} step {} loss {:6f} ".format(epoch, step,loss.item(),))
            t_loss.append(loss.item())

    else:
        # displaying to train loss
        updata_epoch_loss_display( train_loss= t_loss , v_epoch= epoch , envr= "derain train")


def test(test_data_loader, epoch):
    from torchvision.transforms import ToPILImage
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

           # new_loss = torch.mul((1 - criterion_ssim_map(w_out, label)), torch.abs(w_out - label)).mean().cuda()
            Psnr, Ssim = get_psnr_ssim(w_out, label)

            Psnr = round(Psnr.item(), 4)
            Ssim = round(Ssim.item(), 4)

            test_Psnr_sum += Psnr
            test_Ssim_sum += Ssim

            if opt.save_image == True:

               dict_psnr_ssim["Psnr%s_Ssim%s" % (Psnr, Ssim)] = data_path

               out = w_out.cpu().data[0]
               out = ToPILImage()(out)
               image_number = re.findall(r'\d+', data_path[0])[1]
               print(image_number)
               #out.save( "/home/ws/Desktop/derain2020/result_p/%s.jpg" % (image_number))

            # if test_step % opt.test_print_fre == 0:
            #     print("epoch={}  Psnr={}  Ssim={} mseloss{}".format(epoch, Psnr, Ssim, new_loss.item()))
            #     test_Psnr_loss.append(test_Psnr_sum / test_step)
            #     test_Ssim_loss.append(test_Ssim_sum / test_step)

        else:
            #del new_loss
            print("epoch={}  avr_Psnr ={}  avr_Ssim={}".format(epoch, test_Psnr_sum / test_step,
                                                               test_Ssim_sum / test_step))
            write_test_perform("./perform_test.txt", test_Psnr_sum / test_step, test_Ssim_sum / test_step)
            # visdom showing
            print("---->testing over show in visdom")
            display_Psnr_Ssim(Psnr=test_Psnr_sum / test_step, Ssim=test_Ssim_sum / test_step, v_epoch=epoch,
                              env="derain_test")

    print("epoch {} train over-----> save net".format(epoch))
    print("saving checkpoint      save_root{}".format(opt.save_root))
    if os.path.exists(opt.save_root):
        save_checkpoint(root=opt.save_root, model=w, epoch=epoch, model_stage="w_p")

        print("finish save epoch{} checkporint".format({epoch}))
    else:
        raise  Exception("saveroot :{} not found , Checkout it".format(opt.save_root))
    #

    print("all epoch is over ------ ")
    print("show epoch and epoch_loss in visdom")

if __name__ == "__main__":
    os.system('clear')
    main()
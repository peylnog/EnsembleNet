# coding=utf-8
import os
import torch
import argparse
import urllib.request

from Utils.torch_ssim import SSIM
from Utils.ssim_map import SSIM_MAP
from Utils.utils import *
from Utils.Vidsom import *
from Utils.model_init import *
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from MyDataset.Datasets import derain_test_datasets , derain_train_datasets
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, CenterCrop, RandomCrop

from net.model import u_net as wNet

parser = argparse.ArgumentParser(description="PyTorch Derain")
parser.add_argument("--train", default="/home/ws/Desktop/PL/Derain_Dataset2018/train", type=str,
                    help="path to load train datasets(default: none)")
parser.add_argument("--test", default="/home/ws/Desktop/PL/Derain_Dataset2018/test", type=str,
                    help="path to load test datasets(default: none)")
parser.add_argument("--batchSize", type=int, default=32, help="training batch size")
parser.add_argument("--nEpoch", type=int, default=400, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--train_print_freq", type=int, default=100, help="frequency of print train loss on train phase")
parser.add_argument("--test_frequency", type=int, default=1, help="frequency of test")
parser.add_argument("--test_print_freq", type=int, default=200, help="frequency of print train loss on test phase")

parser.add_argument("--cuda",type=str, default="Ture", help="Use cuda?")
parser.add_argument("--gpus", type=int, default=1, help="nums of gpu to use")
parser.add_argument("--startweights", default= 0, type=int, help="start number of net's weight , 0 is None")
parser.add_argument("--initmethod", default='xavier', type=str, help="xavier , kaiming , normal ,orthogonal ,default : xavier")
parser.add_argument("--startepoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--works", type=int, default=8, help="Number of works for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum, Default: 0.9")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained net (default: none)")
parser.add_argument("--saveroot", default="/home/ws/Desktop/derain2020/checkpoints", type=str, help="path to save networks")
parser.add_argument("--report", default=False, type=bool, help="report to wechat")
parser.add_argument("--save_image", default=False, type=bool, help="save test image")
parser.add_argument("--save_image_root", default="", type=str, help="save test image root")



def main():
    global opt, w , criterion_mse , criterion_ssim_map ,criterion_ssim
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

    train_dataset = derain_train_datasets(opt.train, transform=Compose([
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

    w = wNet(init_function) # 1: 256, 2:512
    criterion_mse = nn.MSELoss()
    criterion_ssim_map = SSIM_MAP()
    criterion_ssim = SSIM()

    #print(net)

    # weights start from early
    if opt.startweights:
        if os.path.isfile(opt.saveroot):
            print("=> loading checkpoint '{}'".format(opt.saveroot))
            weights = torch.load(opt.saveroot + '/u/%s.pth'%(opt.startweights-1))
            w.load_state_dict(weights["state_dict"])

        else:
            raise Exception("'{}' is not a file , Check out it again".format(opt.savaroot))


    # optionally copy weights from a checkpoint
    # if opt.pretrained:
    #     if os.path.isfile(opt.pretrained):
    #         print("=> loading net '{}'".format(opt.pretrained))
    #         weights = torch.load(opt.pretrained)
    #         wNet.load_state_dict(weights['state_dict'])
    #     else:
    #         print("=> no net found at '{}'".format(opt.pretrained))

    #if cuda:
    if opt.cuda and torch.cuda.is_available():
        print("==========> Setting GPU")
        w = nn.DataParallel(w, device_ids=[i for i in range(opt.gpus)]).cuda()
        criterion_mse = criterion_mse.cuda()
        criterion_ssim_map = criterion_ssim_map.cuda()
        criterion_ssim = criterion_ssim.cuda()


    else:
        print("==========> Setting CPU")
        w = w.cpu()
        criterion_mse = criterion_mse.cpu()
        criterion_ssim_map = criterion_ssim_map.cpu()

    print("==========> Setting Optimizer")
    optimizer1 = optim.Adam(filter(lambda p: p.requires_grad, w.parameters()), lr=opt.lr)

    print("==========> Training")
    for epoch in range(opt.startepoch, opt.nEpoch + 1):
        train(training_data_loader, optimizer1, epoch)

        if epoch % 100 == 0 :
            opt.lr = 1e-4
            optimizer1 = optim.Adam(filter(lambda p: p.requires_grad, w.parameters()), lr=opt.lr)

        if epoch % opt.test_frequency == 0 :
            test(testing_data_loader ,epoch)

def train(training_data_loader, optimizer, epoch):
    print("training ==========> epoch =", epoch, "lr =", opt.lr)
    w.train()
    # model2.train()
    # model3.train()
    t_loss = []  # save trainloss

    for step, (data, label) in enumerate(training_data_loader, 1):
        if opt.cuda and torch.cuda.is_available():
            data = data.clone().detach().requires_grad_(True).cuda()
            label = label.cuda()
        else:
            data = data.cpu()
            label = label.cpu()

        w.zero_grad()
        optimizer.zero_grad()

        output = w(data)

        mse_loss = criterion_mse(output, label)
        ssim_map = criterion_ssim_map(output, label)
        ssim_loss =  1 - criterion_ssim(output , label)
        #loss = torch.mul((1 - ssim_map) ,torch.abs(output - label)).mean() + 0.01*ssim_loss
        loss = mse_loss

        loss.backward()
        optimizer.step()



        if step % opt.train_print_freq == 0:
            print("epoch{} step {} train_loss {:5f} l1_loss{:6f} ssim_loss{:6f}".format(epoch, step,loss.item(),
                                                                                               mse_loss.item(),
                                                                                               ssim_loss.item()))
            t_loss.append(loss.item())

        del loss, mse_loss, ssim_map
        # displaying to train loss

    updata_epoch_loss_display(train_loss=t_loss, v_epoch=epoch ,envr= 'w train')


def test(test_data_loader, epoch):
    print("------> testing")
    torch.cuda.empty_cache()

    w.eval()
    with torch.no_grad():
        test_Psnr_sum = 0.0
        test_Ssim_sum = 0.0
        # showing list
        test_Psnr_loss = []
        test_Ssim_loss = []
        dict_psnr_ssim = {}
        for test_step, (data, label, _) in enumerate(test_data_loader, 1):
            data = data.cuda()
            label = label.cuda()

            out = w(data)
            del data

            mse_loss = criterion_mse(out, label)
            Psnr, Ssim = get_psnr_ssim(out, label)
            del out
            Psnr = round(Psnr.item(), 5)
            Ssim = round(Ssim.item(), 5)
            # del derain , label
            test_Psnr_sum += Psnr
            test_Ssim_sum += Ssim

            # if opt.save_image :
            #     dict_psnr_ssim["Psnr%s_Ssim%s" % (Psnr, Ssim)] = data_path
            #     out = derain.cpu().data[0]
            #     out = ToPILImage()(out)
            #     image_number = re.findall(r'\d+', data_path[0])[1]
            #     out.save( opt.save_image_root + "/%s_p：%s_s：%s.jpg" % (image_number, Psnr, Ssim))
            if test_step % opt.test_print_freq == 0:
                print("epoch={}  Psnr={}  Ssim={} loss{}".format(epoch, Psnr, Ssim, mse_loss.item()))
                test_Psnr_loss.append(test_Psnr_sum / test_step)
                test_Ssim_loss.append(test_Ssim_sum / test_step)

            del mse_loss,Psnr,Ssim

        else:
            print("epoch={}  avr_Psnr ={}  avr_Ssim={}".format(epoch, test_Psnr_sum / test_step,
                                                               test_Ssim_sum / test_step))
            write_test_perform("/home/ws/Desktop/derain2020/perform_test.txt", test_Psnr_sum / test_step, test_Ssim_sum / test_step)
            # visdom showing
            print("---->testing over show in visdom")
            display_Psnr_Ssim(Psnr=test_Psnr_sum / test_step, Ssim=test_Ssim_sum / test_step, v_epoch=epoch,
                              env="w test")

    print("epoch {} test over-----> save net".format(epoch))
    print("saving checkpoint    save_root{}".format(opt.saveroot))

    #if os.path.isfile(opt.saveroot):
    save_checkpoint(root=opt.saveroot, model=w, epoch=epoch, model_stage="u")
    print("finish save epoch{} checkporint".format({epoch}))
    #else:
    #    raise  Exception("saveroot :{} not found , Checkout it".format(opt.saveroot))

if __name__ == "__main__":
    os.system('clear')
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import *
from Utils.model_init import *
from torchvision import models



class u_net(nn.Module):
    '''u-net get 512 512 3
            out 512 512 3   in order to derain'''

    def __init__(self  ):
        super(u_net,self).__init__()

        self.encode1 = conv_blocks_size_3(in_dim=3 , out_dim=8 ,Use_pool=True)  #256
        self.encode2 = conv_blocks_size_3(in_dim=8 ,out_dim=16,Use_pool=True)   #128
        self.encode3 = conv_blocks_size_3(in_dim=16 ,out_dim=32,Use_pool=True)   #64
        self.encode4 = conv_blocks_size_3(in_dim=32,out_dim=64,Use_pool=True)   #32
        self.encode5 = conv_blocks_size_3(in_dim=64,out_dim=128,Use_pool=True)  #16



        self.decode1 =deconv_blocks_size_3(128,64,Use_pool=True) #32
        self.decode2 = deconv_blocks_size_3(64,32,Use_pool=True) #64
        self.decode3 = deconv_blocks_size_3(32,16,Use_pool=True) #128
        self.decode4 = deconv_blocks_size_3(16,8,True) #256
        self.decode5 = deconv_blocks_size_3(8,3,True) #512

    def forward(self,x):
        """

        :param x:  rain image size: bn channel size
        :return:   same size with x
        """


        encode1 = self.encode1(x)      #256 256 8
        encode2 = self.encode2(encode1) #128  128 16
        encode3 = self.encode3(encode2) #64  64 32
        encode4 = self.encode4(encode3) #32 32 64
        encode5 = self.encode5(encode4) #16 16 128

        decode = self.decode1(encode5)
        decode = F.interpolate(decode, encode4.size()[2:], mode='bilinear', align_corners=True)
        decode = torch.add(decode , encode4) #32

        decode =  self.decode2(decode)
        decode = F.interpolate(decode, encode3.size()[2:], mode='bilinear', align_corners=True)
        decode = torch.add(decode , encode3)#64

        decode = self.decode3(decode)
        decode = F.interpolate(decode, encode2.size()[2:], mode='bilinear', align_corners=True)
        decode = torch.add(decode, encode2)#128

        decode = self.decode4(decode)
        decode = F.interpolate(decode, encode1.size()[2:], mode='bilinear', align_corners=True)
        decode = torch.add(decode, encode1)#256

        decode = self.decode5(decode)
        decode = F.interpolate(decode, x.size()[2:], mode='bilinear', align_corners=True)
        decode = torch.add(decode, x)#512


        return decode



class w_net(nn.Module):
    '''w_net get 512 512 3
             out 512 512 3'''
    def __init__(self ):
        super(w_net, self).__init__()

        self.encode1 = conv_blocks_size_3(in_dim=3, out_dim=8, Use_pool=True )  # 256
        self.encode2 = conv_blocks_size_3(in_dim=8, out_dim=16, Use_pool=True )  # 128
        self.encode3 = conv_blocks_size_3(in_dim=16, out_dim=32, Use_pool=True )  # 64
        self.encode4 = conv_blocks_size_3(in_dim=32, out_dim=64, Use_pool=True )  # 32
        self.encode5 = conv_blocks_size_3(in_dim=64, out_dim=128, Use_pool=True )  # 16

        self.decode2 = deconv_blocks_size_3(128, 64, Use_pool=True)  # 32
        self.decode3 = deconv_blocks_size_3(64, 32, Use_pool=True)  # 64
        self.decode4 = deconv_blocks_size_3(32, 16, Use_pool=True)  # 128
        self.decode5 = deconv_blocks_size_3(16, 8, True)  # 256
        self.decode6 = deconv_blocks_size_3(8, 3, True)  # 512

        self.eencode1 = conv_blocks_size_3(3, 8, True)  # 256
        self.eencode2 = conv_blocks_size_3(8, 16, True)  # 128
        self.eencode3 = conv_blocks_size_3(16, 32, True)  # 64
        self.eencode4 = conv_blocks_size_3(32, 64, True)  # 32
        self.eencode5 = conv_blocks_size_3(64, 128, True)  # 16


        self.ddcode2 = deconv_blocks_size_3(128, 64) #32
        self.ddcode3 = deconv_blocks_size_3(64, 32) #64
        self.ddcode4 = deconv_blocks_size_3(32, 16) #128
        self.ddcode5 = deconv_blocks_size_3(16, 8)  #256
        self.ddcode6 = deconv_blocks_size_3(8, 3)  #512

    def forward(self, x):

        # w-net

        """
        :param x:
        :return:
        """
        encode1 = self.encode1(x)  # 256
        encode2 = self.encode2(encode1)  # 128
        encode3 = self.encode3(encode2)  # 64
        encode4 = self.encode4(encode3)  # 32
        encode5 = self.encode5(encode4)  # 16

        decode = self.decode2(encode5)
        decode = F.interpolate(decode, encode4.size()[2:], mode='bilinear', align_corners=True)
        decode = torch.add(decode, encode4)  # 32

        decode = self.decode3(decode)
        decode = F.interpolate(decode, encode3.size()[2:], mode='bilinear', align_corners=True)
        decode = torch.add(decode, encode3)  # 64

        decode = self.decode4(decode)
        decode = F.interpolate(decode, encode2.size()[2:], mode='bilinear', align_corners=True)
        decode = torch.add(decode, encode2)  # 128

        decode = self.decode5(decode)
        decode = F.interpolate(decode, encode1.size()[2:], mode='bilinear', align_corners=True)
        decode = torch.add(decode, encode1)  # 256

        decode = self.decode6(decode)
        decode = F.interpolate(decode, x.size()[2:], mode='bilinear', align_corners=True)
        decode = torch.add(decode, x)  # 512

        del encode1
        del encode2
        del encode3
        del encode4
        del encode5

        eencode1 = self.eencode1(decode) #256
        eencode2 = self.eencode2(eencode1) #128
        eencode3 = self.eencode3(eencode2) #64
        eencode4 = self.eencode4(eencode3) #32
        eencode5 = self.eencode5(eencode4) #16


        ddecode = self.ddcode2(eencode5) #32
        ddecode = torch.add(eencode4 , F.interpolate(ddecode, eencode4.size()[2:], mode='bilinear', align_corners=True))

        ddecode = self.ddcode3(ddecode) #64
        ddecode = torch.add(eencode3, F.interpolate(ddecode, eencode3.size()[2:], mode='bilinear', align_corners=True))

        ddecode = self.ddcode4(ddecode) #128
        ddecode = torch.add(eencode2, F.interpolate(ddecode, eencode2.size()[2:], mode='bilinear', align_corners=True))

        ddecode = self.ddcode5(ddecode) #256
        ddecode = torch.add(eencode1, F.interpolate(ddecode, eencode1.size()[2:], mode='bilinear', align_corners=True))

        ddecode = self.ddcode6(ddecode) #512
        ddecode = torch.add(decode, F.interpolate(ddecode, decode.size()[2:], mode='bilinear', align_corners=True))

        return ddecode

class res_net(nn.Module):
    def __init__(self ):
        super(res_net, self).__init__()

        self.block1 = res_net_blocks(3,3,ker_size=3,padding=1,stride=1)
        self.block2 = res_net_blocks(3, 3, ker_size=3, padding=1, stride=1)
        self.block3 = res_net_blocks(3, 3, ker_size=3, padding=1, stride=1)
        self.block4 = res_net_blocks(3, 3, ker_size=3, padding=1, stride=1)
        self.block5 = res_net_blocks(3, 3, ker_size=3, padding=1, stride=1)
        self.block6 = res_net_blocks(3, 3, ker_size=3, padding=1, stride=1)
        self.block7 = res_net_blocks(3, 3, ker_size=3, padding=1, stride=1)
        self.block8 = res_net_blocks(3, 3, ker_size=3, padding=1, stride=1)
        self.block9 = res_net_blocks(3, 3, ker_size=3, padding=1, stride=1)
        self.block10 = res_net_blocks(3, 3, ker_size=3, padding=1, stride=1)
        self.block11 = res_net_blocks(3, 3, ker_size=3, padding=1, stride=1)
        self.reduce_channle = res_net_blocks(4, 3, ker_size=3, padding=1, stride=1) #if add insert_img

    def forward(self, x):
        x1 =torch.add ( self.block1(x) ,x )
        del x

        x2 = torch.add ( self.block2(x1) ,x1 )
        del x1

        x3 = torch.add(self.block3(x2), x2)
        del x2

        x4 = torch.add(self.block4(x3) , x3)
        del x3

        x5 = torch.add(self.block5(x4), x4)
        del x4

        x6 = torch.add(self.block6(x5), x5)
        del x5

        x7 = torch.add(self.block7(x6), x6)
        del x6

        x8 = torch.add(self.block8(x7), x7)

        del x7

        x9 = torch.add(self.block9(x8), x8)

        del x8
        x10 = torch.add(self.block10(x9), x9)

        del x9
        x11 = torch.add(self.block11(x10), x10)

        del x10


        return  x11



class Net4(nn.Module):
    def __init__(self):
        super(Net4 , self).__init__()

        self.reduce_channel_1 = conv_block_size_3(9, 6)
        self.reduce_channel_2 = conv_block_size_3(6, 3)

    def forward(self,x1 , x2 , x3):

        x = self.reduce_channel_1(torch.cat([x1,x2,x3] ,dim = 1))
        return self.reduce_channel_2(x)


class Net4_2Net(nn.Module):
    def __init__(self):
        super(Net4_2Net , self).__init__()

        self.reduce_channel_1 = conv_block_size_3(6, 6)
        self.reduce_channel_2 = conv_block_size_3(6, 3)

    def forward(self,x1 ,x2):

        x = self.reduce_channel_1(torch.cat([x1,x2] ,dim = 1))
        return self.reduce_channel_2(x)



class refineNet(nn.Module):

    def __init__(self):
        super(refineNet,self).__init__()

        self.up_dim = deconv_blocks_size_3(12 , 24)
        self.down_dim1 = conv_blocks_size_3(24, 12 , Use_pool= True )

        self.reduce1 = conv_block_size_3(12,6)
        self.reduce2 = conv_block_size_3(6, 3)
    def forward(self, x1 , x2 , x3 , x4):
        x = self.up_dim(torch.cat([x1 , x2 , x3 , x4] , dim = 1))
        x = self.down_dim1(x)

        return self.reduce2(self.reduce1(x))


class refineNet_2Net(nn.Module):

    def __init__(self):
        super(refineNet_2Net,self).__init__()

        self.up_dim = deconv_blocks_size_3(9 , 24)
        self.down_dim1 = conv_blocks_size_3(24, 12 , Use_pool= True )

        self.reduce1 = conv_block_size_3(12,6)
        self.reduce2 = conv_block_size_3(6, 3)
    def forward(self, x1 , x2 , x3 ):
        x = self.up_dim(torch.cat([x1 , x2 , x3 ] , dim = 1))
        x = self.down_dim1(x)

        return self.reduce2(self.reduce1(x))

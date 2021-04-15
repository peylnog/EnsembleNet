# -*- coding: utf-8 -*-

"""
Written by peylnog
2020-3-30
"""
import cv2
import os
from os.path import join
import numpy as np
from VIF import vifp_mscale as vif
from NIQE import niqe
from PIL import Image
from sewar.full_ref import uqi
from sewar.full_ref import vifp

from PSNR_SSIM import get_psnr_ssim
import scipy
from os.path import dirname

if __name__ == '__main__':
    rain_img_root = "/home/ws/Desktop/derain2020/result_new"
    clear_img_root = "/home/ws/Desktop/derain2020/clear"

    AV_VIF = 0
    AV_NIQE = 0
    AV_UQI = 0

    rain_imgs =  [join(rain_img_root , x ) for x in os.listdir(rain_img_root)]
    rain_imgs = sorted(rain_imgs)

    clear_imgs =  [join(clear_img_root , x ) for x in os.listdir(clear_img_root)]
    clear_imgs = sorted(clear_imgs)
    module_path = dirname(__file__)

    params = scipy.io.loadmat(join(module_path, 'niqe_image_params.mat'))

    n = len(rain_imgs)
    for i in range(len(rain_imgs)):
        print(i)
        img_r = rain_imgs[i]
        img_c = clear_imgs[i]
        rain = np.array(Image.open(img_r))
        clear = np.array(Image.open(img_c))

        AV_VIF +=  vifp(  clear , rain)
        AV_UQI +=  uqi(clear,rain)
        rain = np.array(Image.open(img_r).convert('LA'))[:, :, 0]  # ref
        AV_NIQE += niqe(rain , params)

    print("AV_VIF  :"  , AV_VIF/n)
    print("AV_UQI  :"  , AV_UQI/n)
    print("AV_NIQE  :"  , AV_NIQE/n)


#
#
# #------------------------VIF-----------------
#     rain = np.array(Image.open('./demo-images/rain.png'))
#     clear = np.array(Image.open('./demo-images/clear.png'))
#     try:
#         print( " vif : " , vifp(rain , clear) )
#
#     except IOError:
#         print("check out image path")
# # ------------------------niqe-----------------
#
#     try:
#         rain = np.array(Image.open('./demo-images/rain.png').convert('LA'))[:, :, 0]
#         clear = np.array(Image.open('./demo-images/clear.png').convert('LA'))[:, :, 0]
#
#         print( " rain image niqe : " , niqe(rain))
#         print( " clear image niqe : " , niqe(clear))
#     except IOError:
#         print("check out image path")
# # ------------------------psnr-----------------
#     try:
#         rain = np.array(Image.open('./demo-images/rain.png'))
#         clear = np.array(Image.open('./demo-images/clear.png'))
#
#         print( " uqi :  " , uqi(clear,rain))
#     except IOError:
#         print("check out image path")


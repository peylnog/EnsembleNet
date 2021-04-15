# -*- coding: utf-8 -*-

"""
Written by peylnog
2020-3-30
"""
import cv2
import numpy as np
from VIF import vifp_mscale as vif
from NIQE import niqe
from PIL import Image
from sewar.full_ref import uqi
from PSNR_SSIM import get_psnr_ssim


if __name__ == '__main__':
    image_rain =  cv2.imread('./demo-images/rain.png')
    image_clear = cv2.imread('./demo-images/clear.png')
#------------------------VIF-----------------
    try:
        print( " vif : " , vif(image_rain , image_clear) )

    except IOError:
        print("check out image path")
# ------------------------niqe-----------------

    try:
        rain = np.array(Image.open('./demo-images/rain.png').convert('LA'))[:, :, 0]
        clear = np.array(Image.open('./demo-images/clear.png').convert('LA'))[:, :, 0]

        print( " clear image niqe : " , niqe(rain))
        print( " rain image niqe : " , niqe(clear))
    except IOError:
        print("check out image path")
# ------------------------psnr-----------------
    try:
        rain = np.array(Image.open('./demo-images/rain.png'))
        clear = np.array(Image.open('./demo-images/clear.png'))

        print( " uqi :  " , uqi(clear,rain))
    except IOError:
        print("check out image path")


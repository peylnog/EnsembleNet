
import torch
from skimage.measure import compare_psnr , compare_ssim
import numpy as np
from torchvision.transforms import ToPILImage , ToTensor


def get_psnr_ssim(input_img , compared_img , data_range = 1 ,multichannel=True ) :
    '''input and compared should be numpy
    batch_size h w channel'''
#####################torch test with bn ###################
    if isinstance(input_img , torch.Tensor):
        if len(input_img.size()) == 4 :
            input_img = np.squeeze(input_img.cpu().detach().numpy().transpose(0,2,3,1)) #rgb -> gbr




    if isinstance(compared_img , torch.Tensor):
        if len(compared_img.size()) == 4:
            compared_img =np.squeeze(compared_img.cpu().detach().numpy().transpose(0,2,3,1))
#################numpy test#################
    Ssim = compare_ssim(input_img , compared_img ,data_range = data_range ,  multichannel=multichannel)
    Psnr = compare_psnr(input_img , compared_img ,data_range = data_range)
    return Psnr,Ssim



def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr
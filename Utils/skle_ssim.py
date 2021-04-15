from skimage.measure import compare_psnr , compare_ssim
import numpy as np

def get_psnr_ssim(input_img , compared_img) :
    '''input and compared should be numpy
    batch_size 512 512 3'''
    if not type(input_img) == type(np.array([1])):
        input_img = np.squeeze(input_img.cpu().detach().numpy().transpose(0,2,3,1))

    if not type(compared_img) == type(np.array([1])):
        compared_img =np.squeeze(compared_img.cpu().detach().numpy().transpose(0,2,3,1))

    Ssim = compare_ssim(input_img , compared_img ,data_range = 1 ,  multichannel=True)
    Psnr = compare_psnr(input_img , compared_img ,data_range = 1)
    return Psnr,Ssim
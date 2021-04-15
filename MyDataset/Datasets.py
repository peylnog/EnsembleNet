
import re
import torch
import numpy as np
from Utils.torch_ssim import  SSIM
import cv2
import torch.utils.data as Data

from os import listdir
from os.path import join
from PIL import Image
from os.path import basename
from torchvision import transforms as TF
from Utils.utils import get_mean_and_std




def is_image_file(filename):
  filename_lower = filename.lower()
  return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.bmp', '.mat'])


class derain_train_datasets(Data.Dataset):
    '''return rain_img ,clear , classfy_label'''
    def __init__(self, data_root , transform  = None):
        super(derain_train_datasets, self).__init__()

        self.data_filenames = [join(data_root, x) for x in listdir(data_root) if is_image_file(x) and '._' not in x]
        if transform :
            self.transform = transform

    def __getitem__(self, index):
        data_path = self.data_filenames[index]
        data = Image.open(data_path)
        data = self.transform(data)

        label = data[:, :, 512:1024]
        data = data[:, :, :512]

        return data, label

    def __len__(self):
        return len(self.data_filenames)


class derain_test_datasets(Data.Dataset):
    '''return rain_img . classfy_label'''

    def __init__(self, data_root , transform = None):
        super(derain_test_datasets, self).__init__()
        self.data_filenames = [join(data_root, x) for x in listdir(data_root) if is_image_file(x) and '._' not in x]

        if transform:
            self.transform = transform

    def __getitem__(self, index):
        data_path = self.data_filenames[index]
        data = Image.open(data_path)
        data = self.transform(data)

        label = data[:, :, 512:1024]
        data = data[:, :, :512]

        return data, label ,data_path

    def __len__(self):
        return len(self.data_filenames)




class derain_test_datasets_17(Data.Dataset):
    '''return rain_img . classfy_label'''

    def __init__(self, data_root , transform = None):
        super(derain_test_datasets_17, self).__init__()
        self.root = data_root
        rain_root = self.root + '/rain/'
        self.data_filenames = [join(rain_root, x) for x in listdir(rain_root) if is_image_file(x) and '._' not in x]
        self.transform = transform

    def __getitem__(self, index):
        data_path = self.data_filenames[index]
        number = data_path.split('-')[1].split('.')[0]
        label_path = self.root + '/label/norain-' + number + '.png'

        label = Image.open(label_path)
        data = Image.open(data_path)
        if self.transform:
            data = self.transform(data)
            label = self.transform(label)


        return data, label

    def __len__(self):
        return len(self.data_filenames)

import cv2
class derain_train_datasets_17(Data.Dataset):
    '''return rain_img . classfy_label'''

    def __init__(self, data_root , transform = None):
        super(derain_train_datasets_17, self).__init__()
        self.root = data_root
        rain_root = self.root + '/rain/'
        self.data_filenames = [join(rain_root, x) for x in listdir(rain_root) if is_image_file(x) and '._' not in x]
        self.transform = transform

    def __getitem__(self, index):
        data_path = self.data_filenames[index]
        number = data_path.split('-')[1].split('.')[0]
        label_path = self.root + '/label/norain-' + number + '.png'
        ########CV2########
        label = cv2.imread(label_path)[:,:,::-1]
        data = cv2.imread(data_path)[:,:,::-1]
        if data.shape != (481,321,3):
            label = cv2.transpose(label)
            data = cv2.transpose(data)


        label = Image.fromarray(label)
        data = Image.fromarray(data)
        if self.transform:
            data = self.transform(data)
            label = self.transform(label)


        return data, label

    def __len__(self):
        return len(self.data_filenames)

class derain_train_datasets_IC(Data.Dataset):
    '''return rain_img ,clear , classfy_label'''
    def __init__(self, data_root , transform  = None):
        super(derain_train_datasets_IC, self).__init__()

        self.data_filenames = [join(data_root, x) for x in listdir(data_root) if is_image_file(x) and '._' not in x]
        if transform :
            self.transform = transform

    def __getitem__(self, index):
        data_path = self.data_filenames[index]
        data = cv2.imread(data_path)[:, :, ::-1] # bgr to rgb
        #print(data.shape)
        #data = self.transform(data)
        h , w  , c =data.shape
        w = int(w/2)
        label = data[:, :w, :]
        data = data[:, w:, :]
        data = Image.fromarray(data)
        label = Image.fromarray(label)

        return self.transform(data), self.transform(label) , data_path

    def __len__(self):
        return len(self.data_filenames)

class derain_train_datasets_2020(Data.Dataset):
    '''return rain_img ,clear , classfy_label'''
    def __init__(self, p, data_root , transform  = None   ):
        super(derain_train_datasets_2020, self).__init__()

        self.data_filenames = [join(data_root, x) for x in listdir(data_root) if is_image_file(x) and '._' not in x]
        if transform :
            self.transform = transform
        self.p = p
    def __getitem__(self, index):
        data_path = self.data_filenames[index]
        data = Image.open(data_path)
        data = self.transform(data)

        if float(index / len(self))  < self.p :

            label = data[:, :, 512:1024]
            data = data[:, :, :512]
        else:
            label = data[:, :, 512:1024]
            data = label

        return data, label

    def __len__(self):
        return len(self.data_filenames)
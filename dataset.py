# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :mylearn
# @File     :dataset
# @Date     :2021/1/14 21:45
# @Author   :Jay_Lee
# @Software :PyCharm
-------------------------------------------------
"""
import os
import sys
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset

class CIFAR100Train(Dataset):
    """cifar100 train dataset, derived from
    torch.utils.data.DataSet
    """
    def __init__(self,path,transform=None):
        with open(os.path.join(path,'train'),'rb') as cifar100:
            self.data=pickle.load(cifar100,encoding='bytes')
        self.transform=transform
    def __len__(self):
        return len(self.data['fine_labels'.encode()])
    def __getitem__(self, item):
        label = self.data['fine_labels'.encode()][item]
        r = self.data['data'.encode()][item, :1024].reshape(32, 32)
        g = self.data['data'.encode()][item, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][item, 2048:].reshape(32, 32)
        image = np.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, image
class CIFAR100Test(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        with open(os.path.join(path, 'test'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['data'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = np.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, image

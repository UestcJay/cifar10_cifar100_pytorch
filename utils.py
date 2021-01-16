# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :mylearn
# @File     :utils
# @Date     :2021/1/14 20:49
# @Author   :Jay_Lee
# @Software :PyCharm
-------------------------------------------------
"""
import os
import sys
import re
import datetime
import numpy as np

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_network(args):
    if args.v =='cifar100':
        if args.net == 'mobilenet':
            from models.mobilnet import mobilenet
            net = mobilenet()
        elif args.net == 'mobilenetv2':
            from models.mobilenetv2 import mobilenetv2
            net = mobilenetv2()
        else:
            print('the network name you have entered is not supported yet')
            sys.exit()
    elif args.v =='cifar10':
        if args.net == 'mobilenet':
            from models.mobilnet import mobilenet_1
            net = mobilenet_1()
        elif args.net == 'mobilenetv2':
            from models.mobilenetv2 import mobilenetv2_1
            net = mobilenetv2_1()
        else:
            print('the network name you have entered is not supported yet')
            sys.exit()
    if args.gpu:  # use_gpu
        net = net.cuda()
    return net
def get_training_dataloader(mean,std,batch_size=16,num_workers=2,shuffle=True):
    transform_train=transforms.Compose(
        [
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ]
    )
    cifar100_training= torchvision.datasets.CIFAR100(root='./data',train=True,download=True,transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training,shuffle=shuffle,num_workers=num_workers,batch_size=batch_size
    )
    return cifar100_training_loader
def get_test_dataloader(mean,std,batch_size=16,num_workers=2,shuffle=True):
    transform_test=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
    cifar100_test= torchvision.datasets.CIFAR100(root='./data',train=False,download=True,transform=transform_test)
    cifar100_test_loader=DataLoader(
        cifar100_test,shuffle=shuffle,num_workers=num_workers,batch_size=batch_size
    )
    return cifar100_test_loader
def compute_mean_std(cifar100_dataset):
    data_r= np.dstack([cifar100_dataset[i][1][:,:,0] for i in range(len(cifar100_dataset))])
    data_g= np.dstack([cifar100_dataset[i][1][:,:,1] for i in range(len(cifar100_dataset))])
    data_b= np.dstack([cifar100_dataset[i][1][:,:,2] for i in range(len(cifar100_dataset))])

    mean= np.mean(data_r),np.mean(data_g),np.mean(data_b)
    std=np.std(data_r),np.std(data_g),np.std(data_b)

    return mean,std
class WarmUpLR(_LRScheduler):
    def __init__(self,optimizer, total_iters,Last_epoch=-1):
        self.total_iters=total_iters
        super(WarmUpLR,self).__init__(optimizer,Last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
def most_recent_folder(net_weights,fmt):
    folders=os.listdir(net_weights)
    folders = [ f for f in folders if len(os.listdir(os.path.join(net_weights,f)))]
    if len(folders)==0:
        return''
    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]
def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]


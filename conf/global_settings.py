# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :mylearn
# @File     :global_setting
# @Date     :2021/1/14 20:13
# @Author   :Jay_Lee
# @Software :PyCharm
-------------------------------------------------
"""
import os
from datetime import datetime

#CIFAR100 dataset path (python version)
#CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'

#mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
CIFAR10_TRAIN_MEAN =(0.485, 0.456, 0.406)
CIFAR10_TRAIN_STD = (0.229, 0.224, 0.225)

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'
CHECKPOINT_PATH_CIFAR10 = 'checkpoint_cifar10'

#total training epoches
EPOCH = 200
MILESTONES = [60, 120, 160]

EPOCH_CIFAR10 =80
MILESTONES_CIFAR10 = [20, 40, 60]


#initial learning rate
#INIT_LR = 0.1

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#tensorboard log dir
LOG_DIR = 'runs'
LOG_DIR_CIFAR10 = 'runs_cifar10'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10










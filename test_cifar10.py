# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :mylearn
# @File     :test_cifar10
# @Date     :2021/1/16 18:29
# @Author   :Jay_Lee
# @Software :PyCharm
-------------------------------------------------
"""
import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network
from train_cifar10 import get_test_dataloader

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='mobilenet', help='net type')
    parser.add_argument('-v', type=str, default='cifar10', help='dataset type')
    parser.add_argument('-gpu', action='store_true', default=True, help='ues gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batchsize for dataloader')
    parser.add_argument('-weights', type=str,
                        default='./checkpoint_cifar10/mobilenet/Saturday_16_January_2021_05h_11m_00s/mobilenet-165-best.pth',
                        help='the weights file you want to test')
    args = parser.parse_args()

    net=get_network(args)
    cifar10_test_loader = get_test_dataloader(settings.CIFAR10_TRAIN_MEAN
                                               , settings.CIFAR10_TRAIN_STD, num_workers=4, batch_size=args.b)
    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()
    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter,(image,label) in enumerate(cifar10_test_loader):
            print('iteration: {}\ttotal:{}iterations'.format(n_iter+1,len(cifar10_test_loader)))
            if args.gpu:
                image=image.cuda()
                label=label.cuda()
            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            # compute top 5
            correct_5 += correct[:, :5].sum()

            # compute top1
            correct_1 += correct[:, :1].sum()

        print(correct_1 / len(cifar10_test_loader.dataset))
        print("Top 1 err: ", 1 - correct_1 / len(cifar10_test_loader.dataset))
        print("Top 5 err: ", 1 - correct_5 / len(cifar10_test_loader.dataset))
        print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))


# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :mylearn
# @File     :train_cifar10
# @Date     :2021/1/16 17:18
# @Author   :Jay_Lee
# @Software :PyCharm
-------------------------------------------------
"""
import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights


def get_training_dataloader(mean,std,batch_size=16,num_workers=2,shuffle=True):
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )
    cifar10_training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                      transform=transform_train)
    cifar10_training_loader = DataLoader(
        cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size,pin_memory=True
    )
    return cifar10_training_loader
def get_test_dataloader(mean,std,batch_size=16,num_workers=2,shuffle=True):
    transform_test=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
    cifar10_test= torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform_test)
    cifar10_test_loader=DataLoader(
        cifar10_test,shuffle=shuffle,num_workers=num_workers,batch_size=batch_size,pin_memory=True
    )
    return cifar10_test_loader
def train(epoch):
    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(cifar10_training_loader):
        if args.gpu:
            images=images.cuda()
            labels=labels.cuda()
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs,labels)
        loss.backward()
        optimizer.step()

        n_iter= (epoch-1)*len(cifar10_training_loader)+batch_index+1
        last_layer= list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)
        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar10_training_loader.dataset)
        ))
        writer.add_scalar('Train/loss',loss.item(),n_iter)
        if epoch<=args.warm:
            warmup_schedule.step()
    for name,para in net.named_parameters():
        layer,attr=os.path.split(name)
        attr=attr[1:]
        writer.add_histogram("{}/{}".format(layer,attr),para,epoch)
    finish=time.time()
    print('epoch {} consumed training time{:.2f}'.format(epoch,finish-start))
@torch.no_grad()
def eval_training(epoch=0, tb=True):
    start=time.time()
    net.eval()
    test_loss=0.0
    correct=0.0
    for (images,labels) in cifar10_test_loader:
        if args.gpu:
            images=images.cuda()
            labels=labels.cuda()
        outputs=net(images)
        loss= loss_function(outputs,labels)
        test_loss+=loss.item()
        _,preds=outputs.max(1)
        a=preds.eq(labels).sum()
        correct+=a
    finish=time.time()
    # if args.gpu:
    #     print('GPU INFO.....')
    #     print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar10_test_loader.dataset),
        correct.float() / len(cifar10_test_loader.dataset),
        finish - start
    ))
    print()

    # add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar10_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar10_test_loader.dataset), epoch)

    return correct.float() / len(cifar10_test_loader.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='mobilenet', help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='ues gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batchsize for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='init learning rate')
    parser.add_argument('-v', type=str, default='cifar10', help='dataset type')
    args = parser.parse_args()

    net = get_network(args)

    cifar10_training_loader = get_training_dataloader(settings.CIFAR10_TRAIN_MEAN,
                                                       settings.CIFAR10_TRAIN_STD,
                                                       num_workers=4,
                                                       batch_size=args.b,
                                                       shuffle=True)
    cifar10_test_loader = get_test_dataloader(settings.CIFAR10_TRAIN_MEAN,
                                               settings.CIFAR10_TRAIN_STD,
                                               num_workers=4,
                                               batch_size=args.b,
                                               shuffle=True)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES_CIFAR10, gamma=0.2)
    iter_per_epoch = len(cifar10_training_loader)
    warmup_schedule = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH_CIFAR10, args.net, settings.TIME_NOW)
    if not os.path.exists(settings.LOG_DIR_CIFAR10):
        os.mkdir(settings.LOG_DIR_CIFAR10)
    writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR_CIFAR10, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
    best_acc = 0.0
    for epoch in range(1, settings.EPOCH_CIFAR10 + 1):
        if epoch > args.warm:
            train_schedule.step(epoch)
        train(epoch)
        acc = eval_training(epoch)
        if epoch > settings.MILESTONES_CIFAR10[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue
        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('save weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
    writer.close()
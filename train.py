# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :mylearn
# @File     :train
# @Date     :2021/1/15 9:53
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
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights


def train(epoch):
    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if args.gpu:
            images=images.cuda()
            labels=labels.cuda()
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs,labels)
        loss.backward()
        optimizer.step()

        n_iter= (epoch-1)*len(cifar100_training_loader)+batch_index+1
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
            total_samples=len(cifar100_training_loader.dataset)
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
    for (images,labels) in cifar100_test_loader:
        if args.gpu:
            images=images.cuda()
            labels=labels.cuda()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # images = images.to(device)
        # labels = labels.to(device)
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
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    # add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='mobilenet', help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='ues gpu or not')
    parser.add_argument('-v', type=str, default='cifar100', help='dataset type')
    parser.add_argument('-b', type=int, default=128, help='batchsize for dataloader')
    parser.add_argument('-warm', type=int, default=2, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='init learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    net = get_network(args)
    cifar100_training_loader = get_training_dataloader(settings.CIFAR100_TRAIN_MEAN,
                                                       settings.CIFAR100_TRAIN_STD,
                                                       num_workers=4,
                                                       batch_size=args.b,
                                                       shuffle=True)
    cifar100_test_loader = get_test_dataloader(settings.CIFAR100_TRAIN_MEAN,
                                               settings.CIFAR100_TRAIN_STD,
                                               num_workers=4,
                                               batch_size=args.b,
                                               shuffle=True)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)
    iter_per_epoch = len(cifar100_training_loader)
    warmup_schedule = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent foler were found')
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)
    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            best_weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('find the best weight file:{}'.format(best_weights_path))
            print('load the best weight file to test acc...')
            net.load_state_dict(torch.load(best_weights_path))
            best_acc = eval_training(tb=False)
            print('best_acc is %0.2f'.format(best_acc))
        recent_weights_file=most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no such weights file')
        weights_path=os.path.join(settings.settings.CHECKPOINT_PATH, args.net, recent_folder,recent_weights_file)
        print('loading file {} to resume training'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))
        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
    for epoch in range(1,settings.EPOCH+1):
        if epoch>args.warm:
            train_schedule.step(epoch)
        if args.resume:
            if epoch<=resume_epoch:
                continue
        train(epoch)
        acc=eval_training(epoch)
        if epoch>settings.MILESTONES[1] and best_acc<acc:
            weights_path=checkpoint_path.format(net=args.net,epoch=epoch,type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc=acc
            continue
        if not epoch % settings.SAVE_EPOCH:
            weights_path=checkpoint_path.format(net=args.net,epoch=epoch,type='regular')
            print('save weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
    writer.close()

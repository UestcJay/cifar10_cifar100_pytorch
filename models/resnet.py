# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :mylearn
# @File     :resnet
# @Date     :2021/1/17 13:25
# @Author   :Jay_Lee
# @Software :PyCharm
-------------------------------------------------
"""
"""resnet in pytorch

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""
import torch
import torch.nn as nn


class Basic_Block(nn.Module):
    """Basic Block for resnet 18 and resnet 34

      """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(Basic_Block, self).__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * Basic_Block.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * Basic_Block.expansion)
        )
        self.short_cut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * Basic_Block.expansion:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Basic_Block.expansion, 1, stride,  bias=False),
                nn.BatchNorm2d(out_channels * Basic_Block.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.short_cut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        self.residual_function = nn.Conv2d(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion)
        )
        self.short_cut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.short_cut(x))


class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # we use a different inputsize than the original paper
        # so conv2's stride is 1
        self.conv2 = self._make_layer(block, 64, num_block[0], 1)
        self.conv3 = self._make_layer(block, 128, num_block[1], 2)
        self.conv4 = self._make_layer(block, 256, num_block[2], 2)
        self.conv5 = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(Basic_Block, [2, 2, 2, 2])


def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(Basic_Block, [3, 4, 6, 3])


def resnet50():
    return ResNet(BottleNeck, [3, 4, 6, 3])


def resnet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])


def resnet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])

# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :mylearn
# @File     :mobilenetv2
# @Date     :2021/1/14 16:04
# @Author   :Jay_Lee
# @Software :PyCharm
@InProceedings{Sandler_2018_CVPR,
author = {Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
title = {MobileNetV2: Inverted Residuals and Linear Bottlenecks},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
-------------------------------------------------
"""
import math
import torch
import torch.nn as nn


__all__ = ['mobilenetv2']
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv_3x3_bn(inp,oup,stride):
    return nn.Sequential(
        nn.Conv2d(inp,oup,3,stride,1,bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp,oup):
    return nn.Sequential(
        nn.Conv2d(inp,oup,3,1,1,bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup,stride,expand_ratio):
        super(InvertedResidual,self).__init__()
        assert stride in [1,2]
        hidden_dim = round(inp*expand_ratio)
        self.identity = inp==oup and stride==1
        if expand_ratio==1:
            self.conv= nn.Sequential(
                #dw
                nn.Conv2d(inp,hidden_dim,3,stride,1,groups=hidden_dim,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #pw
                nn.Conv2d(hidden_dim,oup,1,1,0,bias=False),
                nn.BatchNorm2d(oup)
            )
        else:
            self.conv=nn.Sequential(
                #pw
                nn.Conv2d(inp,hidden_dim,1,1,0,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #dw
                nn.Conv2d(hidden_dim,hidden_dim,3,stride,1,groups=hidden_dim,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #pw there no relu
                nn.Conv2d(hidden_dim,oup,1,1,0,bias=False),
                nn.BatchNorm2d(oup),
            )
    def forward(self,x):
        if self.identity:
            return x+self.conv(x)
        else:
            return self.conv(x)
class MobileNetV2(nn.Module):
    def __init__(self,num_classes=100, width_mult=1.):
        super(MobileNetV2,self).__init__()
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        block=InvertedResidual
        for t,c,n,s in self.cfgs:
            output_channel= _make_divisible(c*width_mult, 4 if width_mult==0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel,output_channel,s if i==0 else 1,t))
                input_channel=output_channel
        self.features=nn.Sequential(*layers)
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    def forward(self,x):
        x = self.features(x)
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
def mobilenetv2(**kwargs):
    return MobileNetV2(**kwargs)
def mobilenetv2_1(num_classes=10):
    return MobileNetV2(num_classes=num_classes)
if __name__ == '__main__':
    x = torch.rand(1, 3, 224, 224)
    model = mobilenetv2()
    model.eval()
    with torch.no_grad():
        model(x)




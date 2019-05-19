# -*- coding: utf-8 -*-
# Original author: yq_yao
# 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ConvBN(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(ch_out, momentum=0.01, eps=1e-05, affine=True)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.1, inplace=True)

class DarknetBlock(nn.Module):
    def __init__(self, ch_in):
        super().__init__()
        ch_hid = ch_in // 2
        self.conv1 = ConvBN(ch_in, ch_hid, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBN(ch_hid, ch_in, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x

class Darknet19(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.conv = ConvBN(3, 32, kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer1()
        self.layer2 = self._make_layer2()
        self.layer3 = self._make_layer3()
        self.layer4 = self._make_layer4()
        self.layer5 = self._make_layer5()

    def _make_layer1(self):
        layers = [nn.MaxPool2d(kernel_size=2, stride=2),
                    ConvBN(32, 64, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*layers)
        
    def _make_layer2(self):
        layers = [nn.MaxPool2d(kernel_size=2, stride=2),
                  ConvBN(64, 128, kernel_size=3, stride=1, padding=1),
                  ConvBN(128, 64, kernel_size=1, stride=1, padding=1),
                  ConvBN(64, 128, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*layers)

    def _make_layer3(self):
        layers = [nn.MaxPool2d(kernel_size=2, stride=2),
                  ConvBN(128, 256, kernel_size=3, stride=1, padding=1),
                  ConvBN(256, 128, kernel_size=1, stride=1, padding=1),
                  ConvBN(128, 256, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*layers)

    def _make_layer4(self):
        layers = [nn.MaxPool2d(kernel_size=2, stride=2),
                  ConvBN(256, 512, kernel_size=3, stride=1, padding=1),
                  ConvBN(512, 256, kernel_size=1, stride=1, padding=1),
                  ConvBN(256, 512, kernel_size=3, stride=1, padding=1),
                  ConvBN(512, 256, kernel_size=1, stride=1, padding=1),
                  ConvBN(256, 512, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*layers)        

    def _make_layer5(self):
        layers = [nn.MaxPool2d(kernel_size=2, stride=2),
                  ConvBN(512, 1024, kernel_size=3, stride=1, padding=1),
                  ConvBN(1024, 512, kernel_size=1, stride=1, padding=1),
                  ConvBN(512, 1024, kernel_size=3, stride=1, padding=1),
                  ConvBN(1024, 512, kernel_size=1, stride=1, padding=1),
                  ConvBN(512, 1024, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*layers) 

    def forward(self, x):
        out = self.conv(x)

        c1 = self.layer1(out)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        return (c3, c4, c5)


class Darknet53(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        self.conv = ConvBN(3, 32, kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(32, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(256, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(512, num_blocks[4], stride=2)

    def _make_layer(self, ch_in, num_blocks, stride=1):
        layers = [ConvBN(ch_in, ch_in*2, stride=stride, padding=1)]
        for i in range(num_blocks):
            layers.append(DarknetBlock(ch_in * 2))
        return nn.Sequential(*layers) 

    def forward(self, x):
        out = self.conv(x)
        c1 = self.layer1(out)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        return (c3, c4, c5)


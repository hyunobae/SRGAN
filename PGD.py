import argparse

import torch
import torch.nn as nn

import math
import torch
from torch import nn
from module import *


def deepcopy_module(module, target):
    new = nn.Sequential()
    for name, m in module.named_children():
        if name == target:
            new.add_module(name, m)
            new[-1].load_state_dict(m.state_dict())
    return new


def get_name(model):
    names = []
    for key, val in model.state_dict().items():
        name = key.split('.')[0]
        if not name in names:
            names.append(name)

    return names


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        self.config = config
        self.fsize = 64
        self.module_names = []
        self.bz = config.batch_size
        self.is_fade = self.config.is_fade
        self.grow = self.config.grow
        self.dense_layer = None
        self.model = self.get_init_dis()

    def forward(self, x):
        out = self.model(x)
        out = torch.sigmoid(out)
        return out

    def dense_block(self, grow):
        if grow == 0:
            dense = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(64, 128, kernel_size=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 1, kernel_size=1)
            )

        elif grow == 1:
            dense = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(128, 256, kernel_size=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 1, kernel_size=1)
            )

        elif grow == 2:
            dense = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(256, 512, kernel_size=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(512, 1, kernel_size=1)
            )

        elif grow == 3:
            dense = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(512, 1024, kernel_size=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(1024, 1, kernel_size=1)
            )

        return dense

    def first_block(self):
        layers = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
                               nn.LeakyReLU(0.2),
                               nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                               nn.BatchNorm2d(64),
                               nn.LeakyReLU(0.2)
                               )
        return layers

    def get_init_dis(self):
        model = nn.Sequential()
        model.add_module('first_block', self.first_block())
        model.add_module('dense', self.dense_block(self.grow))
        return model

    def freeze_network(self):
        print('freeze network weights')
        for param in self.model.parameters():
            param.requires_grad = False

    def grow_network(self):
        print('grow discriminator and flush dense layer')

        first_block = deepcopy_module(self.model, 'first_block')  # 첫번째 block가져옴

        for name, module in self.model.named_children():  # 나머지 block 가져옴
            if not name == 'first_block' and 'inter_' in name:
                first_block.add_module(name, module)
                first_block[-1].load_state_dict(module.state_dict())

        block = self.add_layer()
        first_block.add_module('inter_'+ str(self.grow), block)
        first_block.add_module('dense', self.dense_block(self.grow))

        self.model = None
        self.model = first_block
        self.is_fade = True

    def intermediate_layer(self):
        in_ = self.fsize
        out_ = in_ * 2

        model = nn.Sequential(
            nn.Conv2d(in_, out_, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_),
            nn.LeakyReLU(0.2),

            nn.Conv2d(out_, out_, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_),
            nn.LeakyReLU(0.2)
        )
        self.fsize = out_
        return model

    def add_layer(self):
        self.grow += 1

        return self.intermediate_layer()

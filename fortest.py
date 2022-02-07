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
    for key, val in model.state_dict().iteritems():
        name = key.split('.')[0]
        if not name in names:
            names.append(name)

    return names


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        self.model = self.get_init_dis()
        self.config = config
        self.fsize = 128
        self.module_names = []
        self.bz = config.batch_size
        self.is_fade = self.config.is_fade
        self.grow = self.config.grow

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(512, 1024, kernel_size=1)
        self.fca = nn.LeakyReLU(0.2)
        self.fc2 = nn.Conv2d(1024, 1, kernel_size=1)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.model(x).view(batch_size)
        out = self.avgpool(out)
        out = self.fc1(out)
        out = self.fca(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

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
        return model

    def freeze_network(self):
        print('freeze network weights')
        for param in self.model.parameters():
            param.requires_grad = False

    def grow_network(self):
        print('grow discriminator')

        first_block = deepcopy_module(self.model, 'first_block') # 첫번째 block가져옴

        new = nn.Sequential()
        new.add_module('first_block', first_block)
        names = get_name(self.model)
        for name, module in self.model.named_children(): # 나머지 block 가져옴
            if not name == 'first_block':
                new.add_module(name, module)
                new[-1].load_state_dict(module.state_dict())

        block = self.add_layer()

        new.add_module('concat', Concatable(new, block))
        # new.add_module('fadein', Fadein(self.config))

        self.model = None
        self.model = new
        self.module_names = get_name(self.model)
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
        add = nn.Sequential()
        add.add_module('inter_'+str(self.grow), self.intermediate_layer())
        self.grow += 1

        return add


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('PGDSRGAN')  # progressive growing discriminator SRGAN
#
#     parser.add_argument('--fsize', default=128, type=int)
#     parser.add_argument('--crop_size', default=96, type=int, help='training images crop size')
#     parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
#                         help='super resolution upscale factor')
#     parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
#     parser.add_argument('--batch_size', default=64, type=int)
#     parser.add_argument('--TICK', type=int, default=1000)
#     parser.add_argument('--trans_tick', type=int, default=200)
#     parser.add_argument('--stablie_tick', type=int, default=100)
#     parser.add_argument('--is_fade', type=bool, default=False)
#     parser.add_argument('--grow', type=int, default=0)
#     opt = parser.parse_args()
#
#     model = Discriminator(opt)
#     print(model)

import torch
import torch.nn as nn

import math
import torch
from torch import nn
from module import *

def deepcopy_module(module, taget):
    new = nn.Sequential()
    for name, m in module.named_children():
        if name == taget:
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
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = self.get_init_dis()
        self.fsize = 128

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

    def get_init_dis(self):
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        return model


    def freeze_network(self, model):
        print('freeze network weights')
        for param in self.model.parameters():
            param.requires_grad = False


    def grow_network(self, size, iter):
        new = nn.Sequential()
        names = get_name(self.model)
        for name, module in self.model.named_children():
            new.add_module(name, module)
            new[-1].load_state_dict(module.state_dict())

        block = self.add_layer(self.fsize)



        self.model = None
        self.model =


    def add_layer(self, fsize):
        in_= fsize
        out_ = fsize*2
        new = nn.Sequential(
            nn.Conv2d(in_, out_, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_),
            nn.LeakyReLU(0.2),

            nn.Conv2d(out_, out_, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_),
            nn.LeakyReLU(0.2)
        )
        self.fsize = out_
        return new
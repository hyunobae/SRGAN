import math
import torch
from torch import nn
from module import Concatable, Fadein


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


class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        self.model = self.get_init_dis()
        self.config = config
        self.fsize = self.config.fsize
        self.module_names = []

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

        new.add_module('concat', Concatable(new, block))
        new.add_module('fadein', Fadein(self.config))

        self.model = None
        self.model = new
        self.module_names = get_name(self.model)

    def add_layer(self, fsize):
        in_ = fsize
        out_ = fsize * 2
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


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

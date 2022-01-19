import torch
import torch.nn as nn


class PixelNormLayer(nn.Module):
    def __init__(self):
        super(PixelNormLayer, self).__init__()

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

    def __repr__(self):
        return self.__class__.__name__


class Concatable(nn.Module):
    def __init__(self, layer1, layer2):
        super(Concatable, self).__init__()
        self.layer1 = layer1
        self.layer2 = layer2

    def forward(self, x):
        y = [self.layer1(x), self.layer2(x)]
        return y


class Fadein(nn.Module):
    def __init__(self, cfg):
        super(Fadein, self).__init__()
        self.alpha = 0.0

    def update_alpha(self, delta):
        self.alpha = self.alpha + delta
        self.alpha = max(0, min(self.alpha, 1.0))

    def forward(self, x):
        return torch.add(x[0].mul(1.0 - self.alpha), x[1].mul(self.alpha))

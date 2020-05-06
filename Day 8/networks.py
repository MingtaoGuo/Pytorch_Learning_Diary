import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
# from torch.nn.utils import spectral_norm


def l2_norm(x):
    return x / (torch.sum(x**2).sqrt() + 1e-10)

class SpectralNorm(nn.Module):
    def __init__(self, out_ch, iter=1):
        super(SpectralNorm, self).__init__()
        self.iter = iter
        self.u = nn.Parameter(torch.randn(out_ch, 1), requires_grad=False).to("cuda:0")
        self.out_ch = out_ch

    def forward(self, W):
        #W: out x in, u: out x 1
        for i in range(self.iter):
            v = l2_norm(torch.matmul(W.transpose(1, 0), self.u))
            self.u = l2_norm(torch.matmul(W, v)).detach()
        sigma = torch.matmul(torch.matmul(self.u.transpose(1, 0), W), v).detach()
        W = W / sigma.clamp_min(1e-10)
        return W


class LinearSN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(LinearSN, self).__init__()
        self.spectral_norm = SpectralNorm(out_ch)
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch))
        init.kaiming_normal_(self.weight)

    def forward(self, x):
        sn_weight = self.spectral_norm(self.weight)
        x = torch.matmul(x, sn_weight.transpose(1, 0))
        return x

class ConvSN(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride, padding):
        super(ConvSN, self).__init__()
        self.spectral_norm = SpectralNorm(out_ch)
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, k_size, k_size))
        init.kaiming_normal_(self.weight)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        size = self.weight.size()
        weight = self.weight.view(size[0], -1)
        sn_weight = self.spectral_norm(weight).view(size[0], size[1], size[2], size[3])
        x = F.conv2d(x, sn_weight, bias=None, stride=self.stride, padding=self.padding)
        return x

class ResBlock_G(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlock_G, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        )
        self.identity = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, 1, 1)
        )

    def forward(self, x):
        temp = self.identity(x)
        x = self.block(x)
        return x + temp

class ResBlock_D_first(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlock_D_first, self).__init__()
        self.block = nn.Sequential(
            ConvSN(in_ch, out_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            ConvSN(out_ch, out_ch, 3, 1, 1),
            nn.AvgPool2d(2, 2)
        )
        self.identity = nn.Sequential(
            nn.AvgPool2d(2, 2),
            ConvSN(in_ch, out_ch, 1, 1, 0)
        )

    def forward(self, x):
        temp = self.identity(x)
        x = self.block(x)
        return x + temp


class ResBlock_D_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlock_D_down, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=False),
            ConvSN(in_ch, out_ch, 3, 1, 1),
            nn.ReLU(inplace=False),
            ConvSN(out_ch, out_ch, 3, 1, 1),
            nn.AvgPool2d(2, 2)
        )
        self.identity = nn.Sequential(
            ConvSN(in_ch, out_ch, 1, 1, 0),
            nn.AvgPool2d(2, 2)
        )

    def forward(self, x):
        temp = self.identity(x)
        x = self.block(x)
        return x + temp

class ResBlock_D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlock_D, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            ConvSN(in_ch, out_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            ConvSN(out_ch, out_ch, 3, 1, 1),
        )

    def forward(self, x):
        temp = x
        x = self.block(x)
        return x + temp

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense = nn.Linear(128, 4*4*256)
        self.blocks = nn.Sequential(
            ResBlock_G(256, 256),
            ResBlock_G(256, 256),
            ResBlock_G(256, 256)
        )
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(256, 3, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dense(x)
        x = x.view(-1, 256, 4, 4)
        x = self.blocks(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.tanh(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.blocks = nn.Sequential(
            ResBlock_D_first(3, 128),
            ResBlock_D_down(128, 128),
            ResBlock_D(128, 128),
            ResBlock_D(128, 128)
        )
        self.relu = nn.ReLU(inplace=True)
        self.dense = LinearSN(128, 1)

    def forward(self, x):
        x = self.blocks(x)
        x = self.relu(x)
        x = torch.sum(x, dim=(2, 3))
        x = x.view(-1, 128)
        x = self.dense(x)
        return x


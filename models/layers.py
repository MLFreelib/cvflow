from collections import OrderedDict

from torch import nn
from torch import sum, log, arange

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


class Conv2dBatchNorm2d(nn.Module):
    def __init__(self, in_channels, out_channels, conv, *args, **kwargs):
        super().__init__()
        self.__layer = nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),
                                                  'bn': nn.BatchNorm2d(out_channels)}))

    def forward(self, x):
        return self.__layer(x)


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))

def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                  padding=pad, stride=stride, bias=False),
        nn.BatchNorm3d(out_planes))

class depthregression(nn.Module):
    def __init__(self, maxdepth):
        super(depthregression, self).__init__()
        self.disp = arange(
            1, 1 + maxdepth, device='cuda', requires_grad=False).float()[None, :, None, None]

    def forward(self, x):
        out = sum(x * self.disp, 1)
        return out


class depthregression_std(nn.Module):
    def __init__(self, maxdepth):
        super(depthregression_std, self).__init__()
        self.disp = arange(
            1, 1 + maxdepth, device='cuda', requires_grad=False).float()[None, :, None, None]

    def forward(self, x, predict):
        disp = (self.disp - predict[:, None, :, :]) ** 2

        out = sum(x * disp, 1)
        out = log(out)
        return out

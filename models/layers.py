from collections import OrderedDict

from torch import nn


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

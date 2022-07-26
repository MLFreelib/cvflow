from collections import OrderedDict
from enum import Enum
from functools import partial

from torch import nn
import torch

from models.layers import Conv2dAuto


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self._block = nn.Identity()

    def forward(self, x):
        return self._block(x)


# Input blocks

class ResNetInputBlock(Block):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU):
        super().__init__(in_channels, out_channels)
        self._block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

# Output blocks

class OutputFormat(Enum):
    CONFIDENCE = 'confidence'
    BBOX = 'bbox'
    SEM_MASK = 'semantic_mask'
    INST_MASK = 'instance_mask'


class OutputBlock(Block):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)


class ClassificationOutput(OutputBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(-1, self.in_channels)
        x = self.decoder(x)
        return {OutputFormat.CONFIDENCE.value: x}


# ResNetBlocks

class ResidualBlock(Block):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, partial(Conv2dAuto,
                                                                                        kernel_size=3,
                                                                                        bias=False)
        self.shortcut = nn.Sequential(OrderedDict(
            {
                'conv': nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                                  stride=self.downsampling, bias=False),
                'bn': nn.BatchNorm2d(self.expanded_channels)

            })) if self.should_apply_shortcut else None

    def conv_bn(self, in_channels, out_channels, conv, *args, **kwargs):
        return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),
                                          'bn': nn.BatchNorm2d(out_channels)}))

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1

    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.relu = activation
        self.blocks = nn.Sequential(
            self.conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            self.relu(),
            self.conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )


class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4

    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, expansion=self.expansion, *args, **kwargs)
        self.blocks = nn.Sequential(
            self.conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
            activation(),
            self.conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
            activation(),
            self.conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )


class ResNetLayer(Block):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1

        self._block = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )


class ResNetBackbone(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """

    def __init__(self, blocks_sizes=(64, 128, 256, 512), deep=(2, 2, 2, 2),
                 activation=nn.ReLU, block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()

        self.blocks_sizes = blocks_sizes

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deep[0], activation=activation,
                        block=block, *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deep[1:])]
        ])

    def forward(self, x):
        for block in self.blocks.children():
            x = block(x)
        return x

# class FPN(Block):
#
#     def __init__(self, in_channels, out_channels, backbone: nn.Module):
#         super().__init__(in_channels, out_channels)
#         self._backbone = backbone
#
#     def forward(self, x):
#         features = list()
#         for block in list(self._backbone.children())[0]:
#             x = block(x)
#             features.append(x)
#
#         results = list()
#         results.append(features[:-1])
#         prev = features[-1]
#         for feature in features[-2::-1]:
#             prev = F.interpolate(prev, size=feature.size()[-2:], mode="nearest")
#             feature = nn.Conv2d(in_channels=feature.size()[1],
#                                 out_channels=prev.size()[1],
#                                 kernel_size=1)(feature)
#
#             prev = prev + feature
#             results.append(prev)
#
#         return results


# YOLO Blocks

class YOLOConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=1, g=1, save_copy=None):
        super().__init__()
        self.save_copy = save_copy
        self.layers = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, bias=False, groups=g),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        out = self.layers(x)
        if self.save_copy is not None:
            self.save_copy.append(out)
        return out

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, downsampling=2):
        super().__init__()
        h = int(c2 / downsampling)  # hidden channels
        self.h = h
        self.c1 = c1
        self.cv1 = YOLOConv(c1, h, 1, 1, p=0)
        self.cv2 = YOLOConv(h, c2, 3, 1, g=g)
        self.layers = nn.Sequential(
            self.cv1,
            self.cv2
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        self.add = True
        if self.add:
            sx = self.cv1(x)
            sx = self.cv2(sx)
            out = x + sx
        else:
            out = self.layers(x)
        return out

class CSPBottleneck(nn.Module):
    def __init__(self, c1, c2, bottlenecks_n=1, downsampling=1, save_copy=None, pre_save=True):
        super().__init__()
        h = int(c2 / downsampling)
        self.cv1 = YOLOConv(c1, h, 1, 1, p=0)
        self.cv2 = YOLOConv(c1, h, 1, 1, p=0)
        self.cv3 = YOLOConv(2 * h, c2, 1, p=0)
        self.save_copy = save_copy
        self.pre_save = pre_save
        self.bn = bottlenecks_n
        self.m = nn.Sequential(*(Bottleneck(h, h, downsampling=1) for _ in range(bottlenecks_n)))

    def forward(self, x):
        x1 = self.cv1(x)
        m = self.m(x1)
        x2 = self.cv2(x)
        x_cat = torch.cat((m, x2), dim=1)
        out = self.cv3(x_cat)
        if self.save_copy is not None:
            if self.pre_save:
                self.save_copy.append(x)
            else:
                self.save_copy.append(out)
        return out

class SPP(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):
        super().__init__()
        h = c1 // 2
        self.cv1 = YOLOConv(c1, h, 1, 1, p=0)
        self.cv2 = YOLOConv(h * 4, c2, 1, 1, p=0)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x1, y1, y2, self.m(y2)], 1))

class YOLOLayer(Block):
    def __init__(self, in_channels, out_channels, k=3, s=2, p=1, bottlenecks_n=3):
        super().__init__(in_channels, out_channels)
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1

        self._block = nn.Sequential(
            YOLOConv(in_channels, out_channels, k=k, s=s, p=p),
            CSPBottleneck(out_channels, out_channels,
                          bottlenecks_n=bottlenecks_n,
                          downsampling=downsampling)
        )

class CSPDarknet(Block):
    def __init__(self, in_channels, out_channels, in_kernel=6, in_stride=2,
                 in_padding=2, blocks_sizes=(64, 128, 256, 512),
                 kernels=(3, 3, 3, 3), strides=(2, 2, 2, 2),
                 paddings=(1, 1, 1, 1), bottlenecks=(3, 6, 9, 3),
                 block=YOLOLayer):
        super().__init__(in_channels, out_channels)
        c2_sizes = list(blocks_sizes[1:])
        c2_sizes.append(out_channels)
        self._block = nn.Sequential(
            YOLOConv(in_channels, blocks_sizes[0], k=in_kernel, s=in_stride, p=in_padding),
            *[block(i[0], i[1], k=i[2], s=i[3], p=i[4], bottlenecks_n=i[5])
              for i in zip(blocks_sizes, c2_sizes, kernels, strides, paddings, bottlenecks)],
            SPP(out_channels, out_channels)
        )

class PANet(nn.Module):
    def __init__(self, blocks_sizes=(64, 128, 256, 512), deep=(2, 2, 2, 2),
                 activation=nn.ReLU, block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
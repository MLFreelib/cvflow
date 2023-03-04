import math
from collections import OrderedDict
from enum import Enum
from functools import partial
from torch.cuda import amp

from torch import nn
import torch

from models.layers import Conv2dAuto
from models.preprocessing import preprocess_for_YOLO
import torch.nn.functional as F

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


class YOLOHead(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=(256, 512, 1024), inplace=True, weight_index=203):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        print(self.no, self.na, self.no * self.na)
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
        self.stride = torch.tensor([8., 16., 32.])
        self.weights = None
        self.weight_index = weight_index
        self.training = False
        self.anchors = torch.tensor([[[1.25000, 1.62500],
                                      [2.00000, 3.75000],
                                      [4.12500, 2.87500]],

                                     [[1.87500, 3.81250],
                                      [3.87500, 2.81250],
                                      [3.68750, 7.43750]],

                                     [[3.62500, 2.81250],
                                      [4.87500, 6.18750],
                                      [11.65625, 10.18750]]])

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))
        out = (torch.cat(z, 1), x)
        return out

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        shape = 1, self.na, ny, nx, 2
        yv, xv = torch.meshgrid(torch.arange(ny, device=d), torch.arange(nx, device=d))
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape).float()
        return grid, anchor_grid

    def import_weights(self, weights_path):
        self.weights = torch.load(weights_path)
        weights_list = [_ for _ in self.weights]
        weight_index = self.weight_index
        self.m[0].weight = nn.Parameter(self.weights[weights_list[weight_index]])
        self.m[0].bias = nn.Parameter(self.weights[weights_list[weight_index + 1]])
        print('SHAPE(0)', self.m[0].weight.shape, self.m[0].bias.shape)
        self.m[1].weight = nn.Parameter(self.weights[weights_list[weight_index + 2]])
        self.m[1].bias = nn.Parameter(self.weights[weights_list[weight_index + 3]])
        print('SHAPE(1)', self.m[1].weight.shape, self.m[1].bias.shape)
        self.m[2].weight = nn.Parameter(self.weights[weights_list[weight_index + 4]])
        self.m[2].bias = nn.Parameter(self.weights[weights_list[weight_index + 5]])
        print('SHAPE(2)', self.m[2].weight.shape, self.m[2].bias.shape)
        weight_index += 6


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

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=1, g=1, save_copy=None):
        super().__init__()
        self.save_copy = save_copy
        # nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),

        self.layers = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, bias=False, groups=g),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        out = self.layers(x)
        if self.save_copy is not None:
            self.save_copy.append(out.clone())
        return out


class YOLOConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=1, g=1, shortcut=None, outs=None):
        super().__init__()
        self.shortcut = shortcut
        self.outs = outs
        self.layers = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, bias=False, groups=g),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        out = self.layers(x)
        if self.shortcut:
            self.outs[self.shortcut] = out.clone()
        return out


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        h = int(c2 * e)  # hidden channels
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
        if self.add:
            sx = self.cv1(x)
            sx = self.cv2(sx)
            out = x + sx
        else:
            out = self.layers(x)
        return out


class CSPBottleneck(nn.Module):
    def __init__(self, c1, c2, bottlenecks_n=1, e=1, save_copy=None, pre_save=True, shortcut=None, outs=None, bs=True):
        super().__init__()
        h = int(c2 * e)
        self.cv1 = Conv(c1, h, 1, 1, p=0)
        self.cv2 = Conv(c1, h, 1, 1, p=0)
        self.cv3 = Conv(2 * h, c2, 1, p=0)
        self.save_copy = save_copy
        self.pre_save = pre_save
        self.shortcut = shortcut
        self.outs = outs
        self.bn = bottlenecks_n
        self.m = nn.Sequential(*(Bottleneck(h, h, e=1, shortcut=bs) for _ in range(bottlenecks_n)))

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
        if self.shortcut:
            self.outs[self.shortcut] = out.clone()
        return out


class SPP(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):
        super().__init__()
        h = c1 // 2
        self.cv1 = Conv(c1, h, 1, 1, p=0)
        self.cv2 = Conv(h * 4, c2, 1, 1, p=0)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x1, y1, y2, self.m(y2)], 1))


class Concat(nn.Module):
    def __init__(self, shortcut, skip_outs, dimension=1):
        super().__init__()
        self.shortcut = shortcut
        self.skip_outs = skip_outs
        self.d = dimension

    def forward(self, x):
        return torch.cat((x, self.skip_outs[self.shortcut]), self.d)


class YOLOLayer(Block):
    def __init__(self, in_channels, out_channels, skip_outs, k=3, s=2, p=1,
                 bottlenecks_n=3, upsample=None, shortcut=None, concat=False, out=None, save_bn=False):
        super().__init__(in_channels, out_channels)
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        layers = []
        if not save_bn:
            conv = YOLOConv(in_channels, out_channels, k=k, s=s, p=p, shortcut=shortcut, outs=out)
        else:
            conv = YOLOConv(in_channels, out_channels, k=k, s=s, p=p)
        layers.append(conv)
        if upsample:
            layers.append(nn.Upsample(scale_factor=upsample, mode='nearest'))
        if concat:
            layers.append(Concat(concat, skip_outs))
        self.shortcut = shortcut
        self.out = out
        self.skip_outs = skip_outs
        cspbn = CSPBottleneck(out_channels, out_channels,
                              bottlenecks_n=bottlenecks_n,
                              e=1 / downsampling)
        if save_bn:
            cspbn.shortcut = shortcut
            cspbn.outs = out
        layers.append(cspbn)
        self._block = nn.Sequential(*layers)


class CSPDarknet(Block):
    def __init__(self, in_channels, out_channels, in_kernel=6, in_stride=2,
                 in_padding=2, blocks_sizes=(64, 128, 256, 512),
                 kernels=(3, 3, 3, 3), strides=(2, 2, 2, 2),
                 paddings=(1, 1, 1, 1), bottlenecks=(3, 6, 9, 3),
                 shortcuts=(None, (2, 0), (3, 0), None),
                 block=YOLOLayer):
        super().__init__(in_channels, out_channels)
        self.skip_outs = {}
        self.weights = None
        self.weight_index = 0
        c2_sizes = list(blocks_sizes[1:])
        c2_sizes.append(out_channels)
        self._block = nn.Sequential(
            YOLOConv(in_channels, blocks_sizes[0], k=in_kernel, s=in_stride, p=in_padding),
            *[YOLOLayer(i[0], i[1], self.skip_outs, k=i[2], s=i[3], p=i[4], bottlenecks_n=i[5], shortcut=i[6],
                        out=self.skip_outs, save_bn=True)
              for i in zip(blocks_sizes, c2_sizes, kernels, strides, paddings, bottlenecks, shortcuts)],
            SPP(out_channels, out_channels)
        )

    def forward(self, x):
        return self._block(x)

    def import_weights(self, weights_path):
        conv_index = 0
        weight_index = 0
        self.weights = torch.load(weights_path)
        weights_list = [_ for _ in self.weights]
        for layer in self._block:
            if isinstance(layer, Conv):
                layer.layers[0].weight = nn.Parameter(self.weights[weights_list[weight_index]])
                layer.layers[0].bias = nn.Parameter(self.weights[weights_list[weight_index + 1]])
                conv_index += 1
                weight_index += 2
            if isinstance(layer, YOLOConv):
                layer.layers[0].weight = nn.Parameter(self.weights[weights_list[weight_index]])
                layer.layers[0].bias = nn.Parameter(self.weights[weights_list[weight_index + 1]])
                conv_index += 1
                weight_index += 2
            if isinstance(layer, SPP):
                layer.cv1.layers[0].weight = nn.Parameter(self.weights[weights_list[weight_index]])
                layer.cv1.layers[0].bias = nn.Parameter(self.weights[weights_list[weight_index + 1]])
                layer.cv2.layers[0].weight = nn.Parameter(self.weights[weights_list[weight_index + 2]])
                layer.cv2.layers[0].bias = nn.Parameter(self.weights[weights_list[weight_index + 3]])
                weight_index += 4
            if isinstance(layer, YOLOLayer):
                for _ in layer._block:
                    if isinstance(_, YOLOConv):

                        _.layers[0].weight = nn.Parameter(self.weights[weights_list[weight_index]])
                        _.layers[0].bias = nn.Parameter(self.weights[weights_list[weight_index + 1]])
                        conv_index += 1
                        weight_index += 2
                    elif isinstance(_, CSPBottleneck):
                        _.cv1.layers[0].weight = nn.Parameter(self.weights[weights_list[weight_index]])
                        _.cv1.layers[0].bias = nn.Parameter(self.weights[weights_list[weight_index + 1]])
                        _.cv2.layers[0].weight = nn.Parameter(self.weights[weights_list[weight_index + 2]])
                        _.cv2.layers[0].bias = nn.Parameter(self.weights[weights_list[weight_index + 3]])
                        _.cv3.layers[0].weight = nn.Parameter(self.weights[weights_list[weight_index + 4]])
                        _.cv3.layers[0].bias = nn.Parameter(self.weights[weights_list[weight_index + 5]])
                        weight_index += 6
                        for i in _.m:
                            i.cv1.layers[0].weight = nn.Parameter(self.weights[weights_list[weight_index]])
                            i.cv1.layers[0].bias = nn.Parameter(self.weights[weights_list[weight_index + 1]])
                            i.cv2.layers[0].weight = nn.Parameter(self.weights[weights_list[weight_index + 2]])
                            i.cv2.layers[0].bias = nn.Parameter(self.weights[weights_list[weight_index + 3]])
                            weight_index += 4
        self.weight_index = weight_index


class PANet(Block):
    def __init__(self, in_channels, out_channels, backbone, bottlenecks_n=3):
        super().__init__(in_channels, out_channels)
        self.outs = backbone.skip_outs
        self.weight_index = backbone.weight_index
        self.bn_outs = []
        self.weights = None
        conv_ins = [(in_channels, in_channels // 2),
                    (in_channels // 2, in_channels // 4),
                    (in_channels // 4, in_channels // 4),
                    (in_channels // 2, in_channels // 2)]
        bn_ins = [(in_channels, in_channels // 2),
                  (in_channels // 2, in_channels // 4),
                  (in_channels // 2, in_channels // 2),
                  (in_channels, in_channels)]
        self.conv1 = YOLOConv(conv_ins[0][0], conv_ins[0][1], p=0, shortcut=(1, 0), outs=self.outs)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.bnc31 = CSPBottleneck(bn_ins[0][0], bn_ins[0][1], e=0.5, bottlenecks_n=bottlenecks_n, bs=False)
        self.conv2 = YOLOConv(conv_ins[1][0], conv_ins[1][1], p=0, shortcut=(4, 0), outs=self.outs)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.bnc32 = CSPBottleneck(bn_ins[1][0], bn_ins[1][1], e=0.5, bottlenecks_n=bottlenecks_n, save_copy=self.bn_outs,
                                   pre_save=False, bs=False)
        self.conv3 = YOLOConv(conv_ins[2][0], conv_ins[2][1], k=3, s=2, p=1)
        self.bnc33 = CSPBottleneck(bn_ins[2][0], bn_ins[2][1], e=0.5, bottlenecks_n=bottlenecks_n, save_copy=self.bn_outs,
                                   pre_save=False, bs=False)
        self.conv4 = YOLOConv(conv_ins[3][0], conv_ins[3][1], k=3, s=2, p=1)
        self.bnc34 = CSPBottleneck(bn_ins[3][0], bn_ins[3][1], e=0.5, bottlenecks_n=bottlenecks_n, save_copy=self.bn_outs,
                                   pre_save=False, bs=False)
        self._block = nn.Sequential(
            self.conv1,
            self.upsample1,
            Concat((3, 0), self.outs),
            self.bnc31,
            self.conv2,
            self.upsample2,
            Concat((2, 0), self.outs),
            self.bnc32,
            self.conv3,
            Concat((4, 0), self.outs),
            self.bnc33,
            self.conv4,
            Concat((1, 0), self.outs),
            self.bnc34
        )

    def forward(self, x):
        self.bn_outs.clear()
        self._block(x)
        return self.bn_outs

    def import_weights(self, weights_path):
        self.weights = torch.load(weights_path)
        weight_index = self.weight_index
        weights_list = [_ for _ in self.weights]
        for layer in self._block:
            if isinstance(layer, YOLOConv):
                layer.layers[0].weight = nn.Parameter(self.weights[weights_list[weight_index]])
                layer.layers[0].bias = nn.Parameter(self.weights[weights_list[weight_index + 1]])
                weight_index += 2
            if isinstance(layer, CSPBottleneck):
                layer.cv1.layers[0].weight = nn.Parameter(self.weights[weights_list[weight_index]])
                layer.cv1.layers[0].bias = nn.Parameter(self.weights[weights_list[weight_index + 1]])
                layer.cv2.layers[0].weight = nn.Parameter(self.weights[weights_list[weight_index + 2]])
                layer.cv2.layers[0].bias = nn.Parameter(self.weights[weights_list[weight_index + 3]])
                layer.cv3.layers[0].weight = nn.Parameter(self.weights[weights_list[weight_index + 4]])
                layer.cv3.layers[0].bias = nn.Parameter(self.weights[weights_list[weight_index + 5]])
                weight_index += 6
                for i in layer.m:
                    i.cv1.layers[0].weight = nn.Parameter(self.weights[weights_list[weight_index]])
                    i.cv1.layers[0].bias = nn.Parameter(self.weights[weights_list[weight_index + 1]])
                    i.cv2.layers[0].weight = nn.Parameter(self.weights[weights_list[weight_index + 2]])
                    i.cv2.layers[0].bias = nn.Parameter(self.weights[weights_list[weight_index + 3]])
                    weight_index += 4
        self.weight_index = weight_index



class CRNN(Block):

    def __init__(self, in_channels, out_channels, img_height, img_width, num_class,
                 map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False):
        super(CRNN, self).__init__(in_channels, out_channels)

        self.cnn, (output_channel, output_height, output_width) = \
            self._cnn_backbone(in_channels, img_height, img_width, leaky_relu)

        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, num_class)

    def _cnn_backbone(self, in_channels, img_height, img_width, leaky_relu):
        assert img_height % 16 == 0
        assert img_width % 4 == 0

        channels = [in_channels, 64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            # shape of input: (batch, input_channel, height, width)
            input_channel = channels[i]
            output_channel = channels[i+1]

            cnn.add_module(
                f'conv{i}',
                nn.Conv2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i])
            )

            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))

            relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}', relu)

        # size of image: (channel, height, width) = (in_channels, img_height, img_width)
        conv_relu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))
        # (64, img_height // 2, img_width // 2)

        conv_relu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))
        # (128, img_height // 4, img_width // 4)

        conv_relu(2)
        conv_relu(3)
        cnn.add_module(
            'pooling2',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (256, img_height // 8, img_width // 4)

        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        cnn.add_module(
            'pooling3',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (512, img_height // 16, img_width // 4)

        conv_relu(6)  # (512, img_height // 16 - 1, img_width // 4 - 1)

        output_channel, output_height, output_width = \
            channels[-1], img_height // 16 - 1, img_width // 4 - 1
        return cnn, (output_channel, output_height, output_width)

    def forward(self, images):
        # shape of images: (batch, channel, height, width)

        conv = self.cnn(images)
        batch, channel, height, width = conv.size()

        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)  # (width, batch, feature)
        seq = self.map_to_seq(conv)

        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)

        output = self.dense(recurrent)
        return output  # shape: (seq_len, batch, num_class)


class MobileV1ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(MobileV1ResidualBlock, self).__init__()

        self.stride = stride
        self.downsample = downsample
        self.blocks = nn.ModuleList([self.convbn_dws(inplanes, planes, 3, stride, pad, dilation),
                                     self.convbn_dws(planes, planes, 3, 1, pad, dilation, second_relu=False)])

    def convbn_dws(self, inp, oup, kernel_size, stride, pad, dilation, second_relu=True):
        if second_relu:
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad,
                          dilation=dilation, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=False)
            )
        else:
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad,
                          dilation=dilation, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )

    def forward(self, x):
        out = x
        for block in self.blocks.children():
            out = block(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class MobileV2ResidualBlock(nn.Module):
    def __init__(self, inp, oup, stride, expanse_ratio, dilation=1):
        super(MobileV2ResidualBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expanse_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        pad = dilation

        if expanse_ratio == 1:
            self.block = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.block = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class MobileStereoFeatureExtractionBlock(nn.Module):
    def __init__(self, add_relus=False):
        super(MobileStereoFeatureExtractionBlock, self).__init__()

        self.expanse_ratio = 3
        self.inplanes = 32
        if add_relus:
            self.firstconv = nn.Sequential(MobileV2ResidualBlock(3, 32, 2, self.expanse_ratio),
                                           nn.ReLU(inplace=True),
                                           MobileV2ResidualBlock(32, 32, 1, self.expanse_ratio),
                                           nn.ReLU(inplace=True),
                                           MobileV2ResidualBlock(32, 32, 1, self.expanse_ratio),
                                           nn.ReLU(inplace=True)
                                           )
        else:
            self.firstconv = nn.Sequential(MobileV2ResidualBlock(3, 32, 2, self.expanse_ratio),
                                           MobileV2ResidualBlock(32, 32, 1, self.expanse_ratio),
                                           MobileV2ResidualBlock(32, 32, 1, self.expanse_ratio)
                                           )

        self.layer1 = self._make_layer(MobileV1ResidualBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(MobileV1ResidualBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(MobileV1ResidualBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(MobileV1ResidualBlock, 128, 3, 1, 1, 2)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = [block(self.inplanes, planes, stride, downsample, pad, dilation)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        feature_volume = torch.cat((l2, l3, l4), dim=1)

        return feature_volume


class MobileStereoNetInputBlock(Block):
    def __init__(self, in_channels=3, out_channels=32):
        super().__init__(in_channels, out_channels)
        self._block = nn.Sequential(
            MobileStereoFeatureExtractionBlock(add_relus=True),
            nn.Sequential(self.convbn(320, 256, 1, 1, 0, 1),
                          nn.ReLU(inplace=True),
                          self.convbn(256, 128, 1, 1, 0, 1),
                          nn.ReLU(inplace=True),
                          self.convbn(128, 64, 1, 1, 0, 1),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(64, 32, 1, 1, 0, 1))
        )

    def convbn(self, in_channels, out_channels, kernel_size, stride, pad, dilation):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        L, R = x
        return self._block(L), self._block(R)


class hourglass2D(nn.Module):
    def __init__(self, in_channels):
        super(hourglass2D, self).__init__()

        self.expanse_ratio = 2

        self.conv1 = MobileV2ResidualBlock(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)

        self.conv2 = MobileV2ResidualBlock(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

        self.conv3 = MobileV2ResidualBlock(in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)

        self.conv4 = MobileV2ResidualBlock(in_channels * 4, in_channels * 4, stride=1, expanse_ratio=self.expanse_ratio)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels))

        self.redir1 = MobileV2ResidualBlock(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
        self.redir2 = MobileV2ResidualBlock(in_channels * 2, in_channels * 2, stride=1,
                                            expanse_ratio=self.expanse_ratio)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6


class MobileStereoNetBackbone(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """

    def __init__(self, maxdisp=192, *args, **kwargs):
        super().__init__()
        self.maxdisp = maxdisp

        self.num_groups = 1

        self.volume_size = 48

        self.hg_size = 48

        self.dres_expanse_ratio = 3

        self.conv3d = nn.Sequential(nn.Conv3d(1, 16, kernel_size=(8, 3, 3), stride=[8, 1, 1], padding=[0, 1, 1]),
                                    nn.BatchNorm3d(16),
                                    nn.ReLU(),
                                    nn.Conv3d(16, 32, kernel_size=(4, 3, 3), stride=[4, 1, 1], padding=[0, 1, 1]),
                                    nn.BatchNorm3d(32),
                                    nn.ReLU(),
                                    nn.Conv3d(32, 16, kernel_size=(2, 3, 3), stride=[2, 1, 1], padding=[0, 1, 1]),
                                    nn.BatchNorm3d(16),
                                    nn.ReLU())

        self.volume11 = nn.Sequential(self.convbn(16, 1, 1, 1, 0, 1),
                                      nn.ReLU(inplace=True))

        self.dres0 = nn.Sequential(MobileV2ResidualBlock(self.volume_size, self.hg_size, 1, self.dres_expanse_ratio),
                                   nn.ReLU(inplace=True),
                                   MobileV2ResidualBlock(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(MobileV2ResidualBlock(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio),
                                   nn.ReLU(inplace=True),
                                   MobileV2ResidualBlock(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio))

        self.encoder_decoder1 = hourglass2D(self.hg_size)

        self.encoder_decoder2 = hourglass2D(self.hg_size)

        self.encoder_decoder3 = hourglass2D(self.hg_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def convbn(self, in_channels, out_channels, kernel_size, stride, pad, dilation):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def interweave_tensors(self, refimg_fea, targetimg_fea):
        B, C, H, W = refimg_fea.shape
        interwoven_features = refimg_fea.new_zeros([B, 2 * C, H, W])
        interwoven_features[:, ::2, :, :] = refimg_fea
        interwoven_features[:, 1::2, :, :] = targetimg_fea
        interwoven_features = interwoven_features.contiguous()
        return interwoven_features

    def forward(self, x):
        featL, featR = x
        B, C, H, W = featL.shape
        volume = featL.new_zeros([B, self.num_groups, self.volume_size, H, W])
        for i in range(self.volume_size):
            if i > 0:
                x = self.interweave_tensors(featL[:, :, :, i:], featR[:, :, :, :-i])
                x = torch.unsqueeze(x, 1)
                x = self.conv3d(x)
                x = torch.squeeze(x, 2)
                x = self.volume11(x)
                volume[:, :, i, :, i:] = x
            else:
                x = self.interweave_tensors(featL, featR)
                x = torch.unsqueeze(x, 1)
                x = self.conv3d(x)
                x = torch.squeeze(x, 2)
                x = self.volume11(x)
                volume[:, :, i, :, :] = x

        volume = volume.contiguous()
        volume = torch.squeeze(volume, 1)

        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.encoder_decoder1(cost0)  # [2, hg_size, 64, 128]
        out2 = self.encoder_decoder2(out1)
        out3 = self.encoder_decoder3(out2)

        return out3


class DepthOutput(OutputBlock):
    def __init__(self, maxdisp=192):
        super().__init__()
        self.hg_size = 48
        self.maxdisp = maxdisp
        self._block = nn.Sequential(self.convbn(self.hg_size, self.hg_size, 3, 1, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1,
                                              bias=False, dilation=1))

    def disparity_regression(self, x, maxdisp):
        assert len(x.shape) == 4
        disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
        disp_values = disp_values.view(1, maxdisp, 1, 1)
        return torch.sum(x * disp_values, 1, keepdim=False)

    def forward(self, x):
        cost3 = self._block(x)
        cost3 = torch.unsqueeze(cost3, 1)
        cost3 = F.interpolate(cost3, [self.maxdisp, x.size()[2], x.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        pred3 = self.disparity_regression(pred3, self.maxdisp)

        return {OutputFormat.DEPTH_MASK.value: pred3}
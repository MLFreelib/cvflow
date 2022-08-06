from collections import OrderedDict
from enum import Enum
from functools import partial
from torch.cuda import amp

from torch import nn
import torch

from models.layers import Conv2dAuto
from models.preprocessing import preprocess_for_YOLO


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
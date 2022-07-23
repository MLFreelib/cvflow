from collections import OrderedDict
from enum import Enum
from functools import partial

from torch import nn

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

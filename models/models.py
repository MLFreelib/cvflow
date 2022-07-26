from typing import Union

from torch import nn

from blocks import Block, ResNetInputBlock, ResNetBackbone, ClassificationOutput, ResNetBottleNeckBlock


class ModelBuilder(nn.Module):
    def __init__(self, input_block: Union[Block, nn.Module],
                 backbone: Union[Block, nn.Module],
                 output_block: Union[Block, nn.Module]):
        super().__init__()
        self.in_block = input_block
        self.backbone = backbone
        self.out_block = output_block

    def forward(self, x):
        x = self.in_block(x)
        x = self.backbone(x)
        x = self.out_block(x)
        return x


# ResNet


def resnet18(in_channels, n_classes):
    return ModelBuilder(
        input_block=ResNetInputBlock(in_channels, 64),
        backbone=ResNetBackbone(deep=(2, 2, 2, 2)),
        output_block=ClassificationOutput(512, n_classes)
    )


def resnet34(in_channels, n_classes):
    return ModelBuilder(
        input_block=ResNetInputBlock(in_channels, 64),
        backbone=ResNetBackbone(deep=(3, 4, 6, 3)),
        output_block=ClassificationOutput(512, n_classes)
    )


def resnet50(in_channels, n_classes):
    return ModelBuilder(
        input_block=ResNetInputBlock(in_channels, 64),
        backbone=ResNetBackbone(deep=(3, 4, 6, 3), block=ResNetBottleNeckBlock),
        output_block=ClassificationOutput(2048, n_classes)
    )


def resnet101(in_channels, n_classes):
    return ModelBuilder(
        input_block=ResNetInputBlock(in_channels, 64),
        backbone=ResNetBackbone(deep=(3, 4, 23, 3), block=ResNetBottleNeckBlock),
        output_block=ClassificationOutput(2048, n_classes)
    )


def resnet152(in_channels, n_classes):
    return ModelBuilder(
        input_block=ResNetInputBlock(in_channels, 64),
        backbone=ResNetBackbone(deep=(3, 8, 36, 3), block=ResNetBottleNeckBlock),
        output_block=ClassificationOutput(4096, n_classes)
    )


def yolo(in_channels):
    return ModelBuilder(
        input_block=,
        backbone=,
        output_block=
    )

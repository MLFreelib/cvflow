from typing import Union

from torch import nn

from models.blocks import *
from models.preprocessing import *
from torch.cuda import amp


class ModelBuilder(nn.Module):
    def __init__(self, input_block: Union[Block, nn.Module],
                 backbone: Union[Block, nn.Module],
                 output_block: Union[Block, nn.Module]):
        super().__init__()
        self.in_block = input_block
        self.backbone = backbone
        self.out_block = output_block
        self.count = 1

    def forward(self, x):
        shape0 = x.shape[2:]
        autocast = False
        with amp.autocast(enabled=autocast):
            x = preprocess_for_YOLO(x, [1, 1, 1])
            shape1 = x.shape[2:]
            self.count += 1
            x = self.in_block(x)
            x = self.backbone(x)
            x = self.out_block(x)
            x = non_max_suppression(x[0])
            scaled_x = scale_coords(shape1, x[0][:, :4], shape0)
            x[0][..., :4] = scaled_x
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


def yolo(in_channels=3, weights_path=None):
    anchors = ((10, 13, 16, 30, 33, 23),
               (30, 61, 62, 45, 59, 119),
               (116, 90, 156, 198, 373, 326))
    input_block = CSPDarknet(in_channels, 1024)
    if weights_path:
        input_block.import_weights(weights_path)
    backbone = PANet(1024, 512, input_block)
    if weights_path:
        backbone.import_weights(weights_path)
    output_block = YOLOHead(anchors=anchors)
    if weights_path:
        output_block.import_weights(weights_path)
    return ModelBuilder(
        input_block=input_block,
        backbone=backbone,
        output_block=output_block
    )
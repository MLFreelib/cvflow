import os
from typing import Union

from ultralytics import YOLO

from models.blocks import *
from models.defects.blocks import OutBlock
from models.defects.ssd300 import SSD300
from models.defects.vgg19 import InputRescale
from models.preprocessing import *
from models.stereo.blocks import GANetInputBlock, GANetBackbone, GANetOutputBlock, DepthOutput, MobileStereoNetBackbone, \
    MobileStereoNetInputBlock


class ModelBuilder(nn.Module):
    def __init__(self, input_block: Union[Block, nn.Module],
                 backbone: Union[Block, nn.Module],
                 output_block: Union[Block, nn.Module],
                 device='cpu'):
        super().__init__()
        self.in_block = input_block
        self.backbone = backbone
        self.out_block = output_block
        self.count = 1
        self.device = device

    def forward(self, x, **kwargs):
        x = self.in_block(x)
        x = self.backbone(x, **kwargs)
        x = self.out_block(x)
        return x


class YOLOBuilder(ModelBuilder):
    def forward(self, imgs):
        autocast = False
        out = []
        for x in imgs:
            with amp.autocast(enabled=autocast):
                shape0 = x.shape[1:]
                x = preprocess_for_YOLO(x, device=device)
                shape1 = x.shape[2:]
                self.count += 1
                x = self.in_block(x)
                x = self.backbone(x)
                x = self.out_block(x)
                x = non_max_suppression(x[0])
                scaled_x = scale_coords(shape1, x[0][:, :4], shape0)
                x[0][..., :4] = scaled_x
                for i in x:
                    out.append({'boxes': i[..., :4],
                                'labels': i[..., 5],
                                'scores': i[..., 4]})
        return out


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


def yolo_large(in_channels=3, weights_path=None):
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
        output_block.import_weights(weights_path=weights_path)
    return YOLOBuilder(
        input_block=input_block,
        backbone=backbone,
        output_block=output_block
    )


def yolo_small(in_channels=3, weights_path=None):
    anchors = ((10, 13, 16, 30, 33, 23),
               (30, 61, 62, 45, 59, 119),
               (116, 90, 156, 198, 373, 326))
    blocks_sizes = (32, 64, 128, 256)
    bottlenecks = (1, 2, 3, 1)
    input_block = CSPDarknet(in_channels, 512, blocks_sizes=blocks_sizes, bottlenecks=bottlenecks)
    if weights_path:
        input_block.import_weights(weights_path)
    backbone = PANet(512, 256, input_block, bottlenecks_n=1)
    if weights_path:
        backbone.import_weights(weights_path)
    output_block = YOLOHead(anchors=anchors, weight_index=backbone.weight_index + 1, ch=(128, 256, 512), nc=1)
    if weights_path:
        output_block.import_weights(weights_path=weights_path)
    return YOLOBuilder(
        input_block=input_block,
        backbone=backbone,
        output_block=output_block
    )

def defects_model(path_to_templates=None, emb_size: int = 512, weights=None, is_train=False, device='cuda'):
    backbone = SSD300(emb_size).to(device)
    if weights is not None:
        checkpoint = torch.load(weights)
        backbone.load_state_dict(checkpoint['model'], strict=False)
    backbone.is_train = is_train
    if not is_train:
        backbone.load_templates(path_to_templates)
        backbone.build_clusters()
    return ModelBuilder(
        input_block=InputRescale(output_size=(300, 300)).to(device),
        backbone=backbone,
        output_block=OutBlock(-1, -1).to(device)
    )

def mobilestereonet(weights = None, is_train=False, device='cuda'):
    input_block = MobileStereoNetInputBlock()
    backbone = MobileStereoNetBackbone(training=is_train)
    output_block = DepthOutput(training=is_train)
    if weights:
        weight_index = input_block.import_weights(weights)
        weight_index = backbone.import_weights(weights, weight_index)
        output_block.import_weights(weights, weight_index)
    model =  ModelBuilder(
        input_block=input_block,
        backbone=backbone,
        output_block=output_block
    )
    torch.nn.DataParallel(model)
    return model

def ganet(weights = None):
    input_block = GANetInputBlock()
    backbone = GANetBackbone()
    output_block = GANetOutputBlock()
    if weights:
        weight_index = input_block.import_weights(weights)
        weight_index = backbone.import_weights(weights, weight_index)
        output_block.import_weights(weights, weight_index)
    model =  ModelBuilder(
        input_block=input_block,
        backbone=backbone,
        output_block=output_block
    )
    torch.nn.DataParallel(model)
    return model


def yolov8(weights=None):
    return ModelBuilder(
        input_block=Block(1, 1),
        backbone=YOLO(weights),
        output_block=Block(1, 1),
    )

def unet(weights=None):
    model = torch.load(weights, map_location=torch.device('cpu'))
    model.eval()
    return ModelBuilder(
        input_block=Block(1, 1),
            backbone=model,
        output_block=Block(1, 1),
    )


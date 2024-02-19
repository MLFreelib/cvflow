import sys
from typing import List

import torch
from torch.optim import Adam

from torch.utils.data import DataLoader
from torchvision import transforms

from components.model_component import ModelDetection, DefectsModel, ModelDepth
from models.losses_structures.stereo_loss import StereoLoss
from models.models import  mobilestereonet
from models.stereo.dataset import StereoDataset

from common.utils import *

import torchvision
from components.muxer_component import DataLoaderMuxer
from components.outer_component import TensorBoardOuter
from components.painter_component import Tiler, BBoxPainter, DepthPainter
from pipeline import Pipeline


def collate_fn(data):
    return data


def get_tiler(name: str, tiler_size: tuple, frame_size: tuple = (640, 1280)) -> Tiler:
    tiler = Tiler(name=name, tiler_size=tiler_size)
    tiler.set_size(frame_size)
    return tiler

def get_depth_model(name: str, model: torch.nn.Module, sources: List[str],
                        transforms: list = None,
                        confidence: float = 0.8) -> ModelDepth:
    model_depth = ModelDepth(name, model)
    [model_depth.add_source(source) for source in sources]
    model_depth.set_transforms(transforms)
    model_depth.set_confidence(conf=confidence)
    return model_depth

if __name__ == '__main__':
    model = mobilestereonet(weights=None,
                           device=get_device(), is_train=True)

    pipeline = Pipeline()
    muxer = DataLoaderMuxer(name='data_muxer')

    transformation =  transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset_left = StereoDataset(data_folder=args['data'], split='train', is_left=True)
    loader_left = DataLoader(dataset=dataset_left, batch_size=2, shuffle=False, collate_fn=collate_fn)
    dataset_right = StereoDataset(data_folder=args['data'], split='train', is_left=False)
    loader_right = DataLoader(dataset=dataset_right, batch_size=2, shuffle=False, collate_fn=collate_fn)
    muxer.add_source(source=loader_left)
    muxer.add_source(source=loader_right)
    model_depth = get_depth_model('depth', model, sources=muxer.get_source_names(),
                                    transforms=[transformation])

    model_depth.is_train = True
    model_depth.add_training_params(
        optimizer=Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999)),
        loss_func=StereoLoss().to(args['device'])
    )
    depth_painter = DepthPainter('depth_painter')
    bbox_painter = BBoxPainter('bboxer')

    outer = TensorBoardOuter('board')
    outer.set_source_names(muxer.get_source_names())
    pipeline.set_device(get_device())
    pipeline.add_all([muxer, model_depth, depth_painter, bbox_painter, outer])
    pipeline.set_max_iters(10_000)
    pipeline.compile()
    pipeline.run()
    pipeline.close()

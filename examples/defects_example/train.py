import sys
from typing import List

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

from components.model_component import ModelDetection, DefectsModel
from models.defects.dataset import DefectsModelDataset
from models.losses_structures.defects_loss import MultiBoxLoss
from models.models import defects_model

sys.path.append('../')

from common.utils import *

import torchvision
from components.muxer_component import DataLoaderMuxer
from components.outer_component import TensorBoardOuter
from components.painter_component import Tiler, BBoxPainter
from pipeline import Pipeline


def collate_fn(data):
    return data


def get_tiler(name: str, tiler_size: tuple, frame_size: tuple = (640, 1280)) -> Tiler:
    tiler = Tiler(name=name, tiler_size=tiler_size)
    tiler.set_size(frame_size)
    return tiler


def get_detection_model(name: str, model: torch.nn.Module, sources: List[str], classes: List[str],
                        transforms: list = None,
                        confidence: float = 0.8) -> ModelDetection:
    model_det = DefectsModel(name, model)
    model_det.set_labels(classes)
    [model_det.add_source(source) for source in sources]
    model_det.set_transforms(transforms)
    model_det.set_confidence(conf=confidence)
    return model_det


if __name__ == '__main__':
    model = defects_model(weights=None,
                          path_to_templates=args['data'], device=get_device(), is_train=True)

    pipeline = Pipeline()
    muxer = DataLoaderMuxer(name='data_muxer')

    transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize((240, 320))
    ])
    dataset = DefectsModelDataset(data_folder=args['data'], split='train', anno_postfix='_anno.txt', img_postfix='.bmp')
    loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    muxer.add_source(source=loader)
    model_det = get_detection_model('detection', model, sources=muxer.get_source_names(),
                                    classes=['Blue_Stain', 'Crack', 'Dead_Knot', 'Knot_missing', 'Live_Knot', 'Marrow',
                                             'Quartzity', 'knot_with_crack', 'overgrown', 'resin'],
                                    transforms=[torchvision.transforms.Resize((300, 300))])

    model_det.is_train = True
    model_det.add_training_params(
        optimizer=SGD(model.parameters(), lr=1e-3),
        loss_func=MultiBoxLoss(priors_cxcy=model.backbone.priors_cxcy).to(args['device'])
    )
    bbox_painter = BBoxPainter('bboxer')

    outer = TensorBoardOuter('board')
    outer.set_source_names(muxer.get_source_names())
    pipeline.set_device(get_device())
    pipeline.add_all([muxer, model_det, bbox_painter, outer])
    pipeline.set_max_iters(10_000)
    pipeline.compile()
    pipeline.run()
    pipeline.close()

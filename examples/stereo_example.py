import sys

sys.path.append('../')

from typing import List
import torch
import torchvision
from common.utils import *
from components.model_component import ModelDepth
from components.muxer_component import SourceMuxer
from components.outer_component import DisplayComponent
from components.painter_component import Tiler, MaskPainter
from components.reader_component import CamReader, VideoReader, ReaderBase, ImageReader
from components.handler_component import Filter
from pipeline import Pipeline

from models.ganet_model import PSMNet as stackhourglass


def get_muxer(readers: List[ReaderBase]) -> SourceMuxer:
    muxer = SourceMuxer('muxer', max_batch_size=1)
    for reader in readers:
        muxer.add_source(reader)
    return muxer


def get_depth_model(name: str, model: torch.nn.Module, sources: List[ReaderBase],
                           transforms: list = None) -> ModelDepth:
    model_depth = ModelDepth(name, model)
    for src in sources:
        model_depth.add_source(src.get_name())
    model_depth.set_transforms(transforms)
    return model_depth


def get_tiler(name: str, tiler_size: tuple, frame_size: tuple = (640, 1280)) -> Tiler:
    tiler = Tiler(name=name, tiler_size=tiler_size)
    tiler.set_size(frame_size)
    return tiler


if __name__ == '__main__':
    model = stackhourglass(maxdisp=192, down=2)
    # checkpoint = torch.load("/content/model_best.pth.tar")
    # checkpoint = torch.load("/Users/s70c3/Projects/cvflow/tests/test_data/ganet.pth.tar")
    # model.load_state_dict(checkpoint['state_dict'], strict=False)
    calib = 1017.

    pipeline = Pipeline()

    readers = []

    image_reader1 = ImageReader('/Users/s70c3/Projects/cvflow/tests/test_data/stereoLeft.png', 'left')
    image_reader2 = ImageReader('/Users/s70c3/Projects/cvflow/tests/test_data/sterepRight.png', 'right')

    image_reader3 = ImageReader('/Users/s70c3/Projects/cvflow/tests/test_data/stereoLeft.png', 'left')
    image_reader4 = ImageReader('/Users/s70c3/Projects/cvflow/tests/test_data/sterepRight.png', 'right')

    readers.append(image_reader1)
    readers.append(image_reader2)
    readers.append(image_reader3)
    readers.append(image_reader4)
    muxer = get_muxer(readers)

    model_depth = get_depth_model('stereo', model, sources=readers)
    model_depth.set_transforms([torchvision.transforms.Resize((960, 544)), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
    tiler = get_tiler('tiler', tiler_size=get_tsize(), frame_size=get_fsize())

    outer = DisplayComponent('display')

    pipeline.set_device('cpu')
    pipeline.add_all([muxer, model_depth,  tiler, outer])
    pipeline.compile()
    pipeline.run()
    pipeline.close()

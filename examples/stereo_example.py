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

def get_usb_cam(path: str, name: str) -> CamReader:
    return CamReader(path, name)


def get_videofile_reader(path: str, name: str) -> VideoReader:
    return VideoReader(path, name)


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
    model = stackhourglass(maxdepth=80, maxdisp=192, down=2)
    checkpoint = torch.load("/Users/s70c3/Projects/cvflow/tests/test_data/ganet.pth.tar", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    calib = 1017

    pipeline = Pipeline()

    readers = []
    usb_srcs = get_cam_srcs()
    for usb_src in usb_srcs:
        readers.append(get_usb_cam(usb_src, usb_src))

    file_srcs = get_video_file_srcs()
    for file_srcs in file_srcs:
        readers.append(get_videofile_reader(file_srcs, file_srcs))

    image_reader1 = ImageReader('/Users/s70c3/Projects/cvflow/tests/test_data/stereoLeft.png', 'left')
    image_reader2 = ImageReader('/Users/s70c3/Projects/cvflow/tests/test_data/sterepRight.png', 'right')

    image_reader3 = ImageReader('/Users/s70c3/Projects/cvflow/tests/test_data/stereoLeft.png', 'left2')
    image_reader4 = ImageReader('/Users/s70c3/Projects/cvflow/tests/test_data/sterepRight.png', 'right2')

    readers.append(image_reader1)
    readers.append(image_reader2)
    readers.append(image_reader3)
    readers.append(image_reader4)
    muxer = get_muxer(readers)

    model_depth = get_depth_model('stereo', model, sources=readers)

    tiler = get_tiler('tiler', tiler_size=get_tsize(), frame_size=get_fsize())

    outer = DisplayComponent('display')

    filter_comp = Filter('filter', ['person'])

    pipeline.set_device(get_device())
    pipeline.add_all([muxer, model_depth, filter_comp,  tiler, outer])
    pipeline.compile()
    pipeline.run()
    pipeline.close()

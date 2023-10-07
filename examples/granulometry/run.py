import sys


sys.path.append('../')
from models.models import unet

from typing import List
import torch
import torchvision
from common.utils import *
from components.model_component import ModelSegmentation
from components.muxer_component import SourceMuxer
from components.outer_component import DisplayComponent
from components.painter_component import Tiler, MaskPainter
from components.reader_component import *
from components.handler_component import Filter, SizeCalculator
from pipeline import Pipeline
from common.utils import argparser as ap

SEM_CLASSES = [
    'froth'
]


def get_usb_cam(path: str, name: str) -> CamReader:
    return CamReader(path, name)


def get_videofile_reader(path: str, name: str) -> VideoReader:
    return VideoReader(path, name)


def get_muxer(readers: List[ReaderBase]) -> SourceMuxer:
    muxer = SourceMuxer('muxer', max_batch_size=1)
    for reader in readers:
        muxer.add_source(reader)
    return muxer


def get_segmentation_model(name: str, model: torch.nn.Module, sources: List[ReaderBase], classes: List[str],
                           transforms: list = None,
                           confidence = .8) -> ModelSegmentation:
    model_segm = ModelSegmentation(name, model, 'unet')
    model_segm.set_labels(classes)
    for src in sources:
        model_segm.add_source(src.get_name())
    model_segm.set_transforms(transforms)
    model_segm.set_confidence(conf=confidence)
    return model_segm


def get_tiler(name: str, tiler_size: tuple, frame_size: tuple = (640, 1280)) -> Tiler:
    tiler = Tiler(name=name, tiler_size=tiler_size)
    tiler.set_size(frame_size)
    return tiler


if __name__ == '__main__':
    args = vars(ap.parse_args())
    model = unet(weights=args['weights'])
    pipeline = Pipeline()

    readers = []
    usb_srcs = get_cam_srcs()
    for usb_src in usb_srcs:
        readers.append(CamReader(usb_src, usb_src))

    name = None
    file_srcs = get_video_file_srcs()
    for i_file_srcs in range(len(file_srcs)):
        name = f'{file_srcs[i_file_srcs]}_{i_file_srcs}'
        readers.append(VideoReader(file_srcs[i_file_srcs], name))

    name = None
    file_srcs = get_img_srcs()
    for i_file_srcs in range(len(file_srcs)):
        name = f'{file_srcs[i_file_srcs]}_{i_file_srcs}'
        readers.append(ImageReader(file_srcs[i_file_srcs], name))

    muxer = get_muxer(readers)
    model_segm = get_segmentation_model('detection', model, sources=readers, classes=SEM_CLASSES,
                                        confidence=get_confidence())

    model_segm.set_transforms([
                               torchvision.transforms.Resize((256, 256)),
                               torchvision.transforms.Grayscale(num_output_channels = 1),
                               ])
    mask_painter = MaskPainter('mask_painter')

    sizer = SizeCalculator('sizer')

    filter_masks = Filter('mask_filter', ['froth'])
    tiler = get_tiler('tiler', tiler_size=get_tsize(), frame_size=get_fsize())

    outer = DisplayComponent('display')
    pipeline.set_device(get_device())
    pipeline.add_all([muxer, model_segm, filter_masks, mask_painter, sizer, tiler, outer])
    pipeline.compile()
    pipeline.run()
    pipeline.close()

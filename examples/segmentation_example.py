import sys

sys.path.append('../')

from typing import List
import torch
import torchvision
from common import utils
from components.model_component import ModelSegmentation
from components.muxer_component import SourceMuxer
from components.outer_component import OuterComponent
from components.painter_component import Tiler, MaskPainter
from components.reader_component import USBCamReader, VideoReader, ReaderBase
from pipeline import Pipeline

SEM_CLASSES = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


def get_usb_cam(path: str, name: str) -> USBCamReader:
    return USBCamReader(path, name)


def get_videofile_reader(path: str, name: str) -> VideoReader:
    return VideoReader(path, name)


def get_muxer(readers: List[ReaderBase]) -> SourceMuxer:
    muxer = SourceMuxer('muxer', max_batch_size=1)
    for reader in readers:
        muxer.add_source(reader)
    return muxer


def get_segmentation_model(name: str, model: torch.nn.Module, sources: List[ReaderBase], classes: List[str],
                           transforms: list = None,
                           confidence: float = 0.8) -> ModelSegmentation:
    model_segm = ModelSegmentation(name, model)
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
    args = utils.args
    model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
    pipeline = Pipeline()

    readers = []
    usb_srcs = args['usbcam']
    if usb_srcs is not None:
        usb_srcs = usb_srcs.split(',')
        for usb_src in usb_srcs:
            readers.append(get_usb_cam(usb_src, usb_src))

    file_srcs = args['videofile']
    if file_srcs is not None:
        file_srcs = file_srcs.split(',')
        for file_srcs in file_srcs:
            readers.append(get_videofile_reader(file_srcs, file_srcs))

    muxer = get_muxer(readers)

    model_segm = get_segmentation_model('detection', model, sources=readers, classes=SEM_CLASSES, confidence=0.5)

    model_segm.set_transforms([torchvision.transforms.Resize((480, 640))])
    mask_painter = MaskPainter('mask_painter')

    resolution = [int(v) for v in args['tsize'].split(',')]
    tiler = get_tiler('tiler', tiler_size=(2, 2), frame_size=tuple(resolution))

    outer = OuterComponent('display', ['tiler'])

    pipeline.set_device('cuda')
    pipeline.add_all([muxer, model_segm, mask_painter, tiler, outer])
    pipeline.compile()
    pipeline.run()
    pipeline.close()

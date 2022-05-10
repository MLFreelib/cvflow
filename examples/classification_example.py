import sys

sys.path.append('../')

from typing import List
import torch
import torchvision
import pandas as pd
from common.utils import *
from components.model_component import ModelClassification
from components.muxer_component import SourceMuxer
from components.outer_component import OuterComponent
from components.painter_component import Tiler, LabelPainter
from components.reader_component import USBCamReader, VideoReader, ReaderBase
from components.handler_component import Filter

from pipeline import Pipeline


def get_usb_cam(path: str, name: str) -> USBCamReader:
    return USBCamReader(path, name)


def get_videofile_reader(path: str, name: str) -> VideoReader:
    return VideoReader(path, name)


def get_muxer(readers: List[ReaderBase]) -> SourceMuxer:
    muxer = SourceMuxer('muxer', max_batch_size=1)
    for reader in readers:
        muxer.add_source(reader)
    return muxer


def get_classification_model(name: str, model: torch.nn.Module, sources: List[ReaderBase], classes: List[str],
                             transforms: list = None,
                             confidence: float = 0.8) -> ModelClassification:
    model_class = ModelClassification(name, model)
    model_class.set_labels(classes)
    for src in sources:
        model_class.add_source(src.get_name())
    model_class.set_transforms(transforms)
    model_class.set_confidence(conf=confidence)
    return model_class


def get_tiler(name: str, tiler_size: tuple, frame_size: tuple = (640, 1280)) -> Tiler:
    tiler = Tiler(name=name, tiler_size=tiler_size)
    tiler.set_size(frame_size)
    return tiler


def get_labels(path: str):
    df_labels = pd.read_csv(path, sep=',')
    df_labels['label'] = df_labels.label.str.strip()
    return df_labels.label


if __name__ == '__main__':
    model = torchvision.models.resnet50(pretrained=True)
    labels = get_labels('ImageNetClasses.csv')
    pipeline = Pipeline()

    readers = []
    usb_srcs = get_cam_srcs()
    for usb_src in usb_srcs:
        readers.append(get_usb_cam(usb_src, usb_src))

    file_srcs = get_video_file_srcs()
    for file_srcs in file_srcs:
        readers.append(get_videofile_reader(file_srcs, file_srcs))

    muxer = get_muxer(readers)
    model_class = get_classification_model('classification', model, sources=readers, classes=list(labels.values))

    model_class.set_transforms([torchvision.transforms.Resize((240, 320))])
    label_painter = LabelPainter('lblpainter')

    tiler = get_tiler('tiler', tiler_size=get_tsize(), frame_size=get_fsize())

    outer = OuterComponent('display', ['tiler'])

    pipeline.set_device(get_device())
    pipeline.add_all([muxer, model_class, label_painter, tiler, outer])
    pipeline.compile()
    pipeline.run()
    pipeline.close()

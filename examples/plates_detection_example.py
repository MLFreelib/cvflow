import sys

sys.path.append('../')
sys.path.append('./yolov5')

from common.utils import *

from typing import List
import torch
import torchvision
from components.model_component import ModelDetection
from components.muxer_component import SourceMuxer
from components.outer_component import DisplayComponent
from components.painter_component import Tiler, BBoxPainter
from components.reader_component import CamReader, VideoReader, ReaderBase, ImageReader
from components.handler_component import Filter, Counter

from pipeline import Pipeline
from models.plates_model import PlatesModel

def get_usb_cam(path: str, name: str) -> CamReader:
    return CamReader(path, name)


def get_videofile_reader(path: str, name: str) -> VideoReader:
    return VideoReader(path, name)


def get_muxer(readers: List[ReaderBase]) -> SourceMuxer:
    muxer = SourceMuxer('muxer', max_batch_size=1)
    for reader in readers:
        muxer.add_source(reader)
    return muxer


def get_detection_model(name: str, model: torch.nn.Module, sources: List[ReaderBase], classes: List[str],
                        transforms: list = None,
                        confidence: float = 0.8) -> ModelDetection:
    model_det = ModelDetection(name, model)
    #model_det.set_labels(classes)
    for src in sources:
        model_det.add_source(src.get_name())
    model_det.set_transforms(transforms)
    model_det.set_confidence(conf=confidence)
    return model_det


def get_tiler(name: str, tiler_size: tuple, frame_size: tuple = (640, 1280)) -> Tiler:
    tiler = Tiler(name=name, tiler_size=tiler_size)
    tiler.set_size(frame_size)
    return tiler


if __name__ == '__main__':
    model = PlatesModel()
    pipeline = Pipeline()

    readers = []
    usb_srcs = get_cam_srcs()
    for usb_src in usb_srcs:
        readers.append(get_usb_cam(usb_src, usb_src))

    name = None
    file_srcs = get_video_file_srcs()
    for i_file_srcs in range(len(file_srcs)):
        name = f'{file_srcs[i_file_srcs]}_{i_file_srcs}'
        readers.append(get_videofile_reader(file_srcs[i_file_srcs], name))

    image_reader1 = ImageReader('E:\PyCharmProjects\cvflow\\tests\\test_data\zebra.jpg', 'zebra1')
    image_reader2 = ImageReader('E:\PyCharmProjects\cvflow\\tests\\test_data\zebra.jpg', 'zebra2')

    readers.append(image_reader1)
    readers.append(image_reader2)
    muxer = get_muxer(readers)
    model_det = get_detection_model('detection', model, sources=readers, classes=COCO_INSTANCE_CATEGORY_NAMES)

    model_det.set_transforms([torchvision.transforms.Resize((240, 320))])
    model_det.set_source_names([f'zebra2'])
    bbox_painter = BBoxPainter('bboxer', font_path=get_font())

    tiler = get_tiler('tiler', tiler_size=get_tsize(), frame_size=get_fsize())

    outer = DisplayComponent('display')
    pipeline.set_device(get_device())
    pipeline.add_all([muxer, model_det, bbox_painter, tiler, outer])
    pipeline.compile()
    pipeline.run()
    pipeline.close()
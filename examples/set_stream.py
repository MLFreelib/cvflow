import sys

import configparser

from typing import List
import torch
from components.model_component import ModelDetection, ModelDetectionForOC
from components.muxer_component import SourceMuxer
from components.handler_component import Counter, Filter
from components.painter_component import Tiler, BBoxPainter
from components.reader_component import CamReader, VideoReader, ReaderBase
#from components.tracker_component import CorrelationBasedTrackerComponent

COCO_INSTANCE_CATEGORY_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                                'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
                                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                                'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
                                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                                'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
                                'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                                'cell phone',
                                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
                                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
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


def get_detection_model(name: str, model: torch.nn.Module, sources: List[ReaderBase], classes: List[str],
                        transforms: list = None,
                        confidence: float = 0.5) -> ModelDetection:
    model_det = ModelDetection(name, model)
    model_det.set_labels(classes)
    for src in sources:
        model_det.add_source(src.get_name())
    model_det.set_transforms(transforms)
    model_det.set_confidence(conf=confidence)
    return model_det

def get_oc_model(name: str, model: torch.nn.Module, sources: List[ReaderBase], classes: List[str],
                        transforms: list = None,
                        confidence: float = 0.5) -> ModelDetection:
    model_det = ModelDetectionForOC(name, model)
    model_det.set_labels(classes)
    for src in sources:
        model_det.add_source(src.get_name())
    model_det.set_transforms(transforms)
    model_det.set_confidence(conf=confidence)
    return model_det


# def get_tracker(name: str, model: torch.nn.Module, sources: List[ReaderBase],
#                 classes: List[str], lines=None) -> CorrelationBasedTrackerComponent:
#     tracker = CorrelationBasedTrackerComponent(name)
#     tracker.set_labels(classes)
#     for src in sources:
#         tracker.add_source(src.get_name())
#     return tracker


def get_counter(name: str, lines) -> Counter:
    counter = Counter(name, lines)
    return counter


def get_tiler(name: str, tiler_size: tuple, frame_size: tuple = (640, 1280)) -> Tiler:
    tiler = Tiler(name=name, tiler_size=tiler_size)
    tiler.set_size(frame_size)
    return tiler

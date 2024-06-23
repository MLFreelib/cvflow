import configparser
import os
import sys

from typing import List
import torch
import torchvision
from common.utils import *
from components.model_component import ModelDepth
from components.handler_component import DistanceCalculator
from components.muxer_component import SourceMuxer
from components.outer_component import DisplayComponent, FileWriterComponent
from components.tracker_component import ManualROICorrelationBasedTracker
from components.painter_component import Tiler, DepthPainter, BBoxPainter
from components.reader_component import ReaderBase, ImageReader, VideoReader, CamReader
from pipeline import Pipeline
from common.utils import argparser as ap

from models.models import mobilestereonet
from models.models import ganet

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


def get_tracker(name: str, sources: List[ReaderBase],
                classes: List[str], boxes=None, tracker_type = None) -> ManualROICorrelationBasedTracker:
    tracker = ManualROICorrelationBasedTracker(name, boxes, tracker_type)
    tracker.set_labels(classes)
    for src in sources:
        tracker.add_source(src.get_name())
    return tracker


def get_tiler(name: str, tiler_size: tuple, frame_size: tuple = (640, 1280)) -> Tiler:
    tiler = Tiler(name=name, tiler_size=tiler_size)
    tiler.set_size(frame_size)
    return tiler


def get_usb_cam(path: str, name: str) -> CamReader:
    return CamReader(path, name)


def get_videofile_reader(path: str, name: str) -> VideoReader:
    return VideoReader(path, name)


if __name__ == '__main__':
    args = vars(ap.parse_args())
    checkpoint = torch.load(args['weights'],
                            map_location=torch.device(get_device()))['model']
    model = mobilestereonet(checkpoint, device=get_device())
    if get_device() == 'cuda':
        model = torch.nn.DataParallel(model)
    pipeline = Pipeline()

    readers = []

    usb_srcs = get_cam_srcs()
    for usb_src in usb_srcs:
        readers.append(get_usb_cam(usb_src, usb_src))

    file_srcs = get_video_file_srcs()
    for file_src in file_srcs:
        readers.append(get_videofile_reader(file_src, os.path.basename(file_src)))

    name = None
    file_srcs = get_img_srcs()
    for i_file_srcs in range(len(file_srcs)):
        name = f'{file_srcs[i_file_srcs]}_{i_file_srcs}'
        readers.append(ImageReader(file_srcs[i_file_srcs], name))


    config = configparser.ConfigParser()
    config.read(args['config'])
    bboxes_list = eval(config.get('bboxes', 'values'))
    bboxes = []
    for bb in bboxes_list:
        bboxes.append((bb['points'][0][0], bb['points'][0][1],
                       bb['points'][0][0] + bb['points'][1][0],
                       bb['points'][0][1] + bb['points'][1][1]))

    muxer = get_muxer(readers)

    model_depth = get_depth_model('stereo', model, sources=readers)

    model_depth.set_transforms(
        [torchvision.transforms.Resize((512, 960)), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                     std=[0.229, 0.224, 0.225])])
    tracker = get_tracker('tracking', sources=readers, classes=["object"], boxes=bboxes, tracker_type='MEDIANFLOW')
    dist = DistanceCalculator('distance')

    depth_painter = DepthPainter('depth_painter')
    bbox_painter = BBoxPainter('bboxer')
    tiler = get_tiler('tiler', tiler_size=get_tsize(), frame_size=get_fsize())

    outer = DisplayComponent('file')

    pipeline.set_device(get_device())
    #
    pipeline.add_all([muxer,  model_depth, tracker,  dist,  bbox_painter, depth_painter, tiler, outer])
    pipeline.compile()
    pipeline.run()
    pipeline.close()

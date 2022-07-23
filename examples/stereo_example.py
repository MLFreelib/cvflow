import configparser
import os
import sys

sys.path.append('../')

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
from components.reader_component import ReaderBase, ImageReader, VideoReader
from pipeline import Pipeline

# from models.ganet_model import PSMNet as depth_model
from models.mobilestereonet_model import MSNet2D as depth_model


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
                classes: List[str], boxes=None) -> ManualROICorrelationBasedTracker:
    tracker = ManualROICorrelationBasedTracker(name, boxes)
    tracker.set_labels(classes)
    for src in sources:
        tracker.add_source(src.get_name())
    return tracker


def get_tiler(name: str, tiler_size: tuple, frame_size: tuple = (640, 1280)) -> Tiler:
    tiler = Tiler(name=name, tiler_size=tiler_size)
    tiler.set_size(frame_size)
    return tiler


if __name__ == '__main__':
    model = depth_model()
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(os.path.join(os.path.dirname(__file__), '..', 'tests', 'test_data', 'best.ckpt'))
    model.load_state_dict(checkpoint['model'], strict=False)

    pipeline = Pipeline()

    readers = []

    image_reader1 = VideoReader(os.path.join(os.path.dirname(__file__), '..', 'tests', 'test_data', 'top_l.mov'), 'left')
    image_reader2 = VideoReader(os.path.join(os.path.dirname(__file__), '..', 'tests', 'test_data', 'top_r.mov'),
                                'right')

    # image_reader1 = ImageReader("/content/cvflow/70_50_1l.png", "left")
    # image_reader2 = ImageReader("/content/cvflow/70_50_1r.png", "right")
    readers.append(image_reader1)
    readers.append(image_reader2)

    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), '..', 'tests', 'test_data', 'conf.txt'))
    bboxes_list = eval(config.get('bboxes', 'values'))
    bboxes = []
    for bb in bboxes_list:
        bboxes.append((bb['points'][0][0], bb['points'][0][1],
                       bb['points'][0][0] + bb['points'][1][0],
                       bb['points'][0][1] + bb['points'][1][1]))

    muxer = get_muxer(readers)

    model_depth = get_depth_model('stereo', model, sources=readers)
    #
    model_depth.set_transforms(
        [torchvision.transforms.Resize((512, 960)), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                     std=[0.229, 0.224, 0.225])])
    tracker = get_tracker('tracking', sources=readers, classes=["object"], boxes=bboxes)
    dist = DistanceCalculator('distance')

    depth_painter = DepthPainter('depth_painter')
    bbox_painter = BBoxPainter('bboxer', font_path=os.path.join(os.path.dirname(__file__), '..', 'fonts', "OpenSans-VariableFont_wdth,wght.ttf"))
    tiler = get_tiler('tiler', tiler_size=get_tsize(), frame_size=get_fsize())

    outer = FileWriterComponent('file', 'file.avi')

    pipeline.set_device('cuda')
    pipeline.add_all([muxer, model_depth, depth_painter, tiler, outer])
    pipeline.compile()
    pipeline.run()
    pipeline.close()

import sys

sys.path.append('../')

from typing import List
import torch
import torchvision
from common import utils
from components.ModelComponent import ModelDetection
from components.MuxerComponent import SourceMuxer
from components.OuterComponent import OuterComponent
from components.PainterComponent import Tiler, BBoxPainter
from components.ReaderComponent import USBCamReader, VideoReader, ReaderBase
from pipeline import Pipeline

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
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


def get_detection_model(name: str, model: torch.nn.Module, sources: List[ReaderBase], classes: List[str],
                        transforms: list = None,
                        confidence: float = 0.8) -> ModelDetection:
    model_det = ModelDetection(name, model)
    model_det.set_labels(classes)
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
    args = utils.args
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
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
    model_det = get_detection_model('detection', model, sources=readers, classes=COCO_INSTANCE_CATEGORY_NAMES)

    model_det.set_transforms([torchvision.transforms.Resize((480, 640))])
    bbox_painter = BBoxPainter('bboxer', font_path=args['font'])

    resolution = [int(v) for v in args['tsize'].split(',')]
    tiler = get_tiler('tiler', tiler_size=(2, 2), frame_size=tuple(resolution))

    outer = OuterComponent('display', ['tiler'])

    pipeline.set_device('cuda')
    pipeline.add_all([muxer, model_det, bbox_painter, tiler, outer])
    pipeline.compile()
    pipeline.run()
    pipeline.close()

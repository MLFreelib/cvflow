
from common.utils import *

import torch
import torchvision
from components.model_component import ModelDetection
from components.muxer_component import SourceMuxer
from components.outer_component import DisplayComponent
from components.painter_component import Tiler, BBoxPainter
from components.reader_component import *
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
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
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
    model_det = get_detection_model('detection', model, sources=readers, classes=COCO_INSTANCE_CATEGORY_NAMES)

    model_det.set_transforms([torchvision.transforms.Resize((240, 320))])
    model_det.set_source_names([reader.get_name() for reader in readers])
    bbox_painter = BBoxPainter('bboxer')

    tiler = get_tiler('tiler', tiler_size=get_tsize(), frame_size=get_fsize())

    outer = DisplayComponent('display')
    pipeline.set_device(get_device())
    pipeline.add_all([muxer, model_det, bbox_painter, tiler, outer])
    pipeline.compile()
    pipeline.run()
    pipeline.close()

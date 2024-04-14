import sys

from components.outer_component import DisplayComponent
from components.painter_component import Tiler, BBoxPainter
from components.reader_component import *
from pipeline import Pipeline
from examples.set_stream import *
from common.utils import *
from models.models import yolov8

if __name__ == '__main__':
    weights_path = get_weights()
    device = get_device()
    model = yolov8(weights=weights_path, device=device)

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

    muxer = get_muxer(readers)
    model_det = get_detection_model('detection', model, sources=readers, classes=COCO_INSTANCE_CATEGORY_NAMES)
    lines = get_line()
    line = [lines[0][0][0], lines[0][0][1], lines[0][1][0], lines[0][1][1]]
    counter = get_counter('counter', lines)
    bbox_painter = BBoxPainter('bboxer')

    tiler = get_tiler('tiler', tiler_size=(2, 2), frame_size=(1440, 2160))

    outer = DisplayComponent('display')

    pipeline.set_device('cpu')
    pipeline.add_all([muxer, model_det, counter, bbox_painter, tiler, outer])
    pipeline.compile()
    pipeline.run()

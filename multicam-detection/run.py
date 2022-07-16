import sys

import configparser

sys.path.append('../')
sys.path.append('backend')

import torchvision
from components.outer_component import DisplayComponent
from components.painter_component import Tiler, BBoxPainter
from pipeline import Pipeline
from set_stream import *
from common.utils import *
from yolo import YOLO

args = get_args()

model = YOLO(calib_path=args['calibration'].split(','))
device = 'cpu'

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

bbox_painter = BBoxPainter('bboxer', font_path=args['font'])

tiler = get_tiler('tiler', tiler_size=(2, 2), frame_size=(1440, 2160))
tiler.set_source_names(['plot', ])

outer = DisplayComponent('display')

pipeline.set_device('cpu')
pipeline.add_all([muxer, model_det, bbox_painter, tiler, outer])
pipeline.compile()
pipeline.run()
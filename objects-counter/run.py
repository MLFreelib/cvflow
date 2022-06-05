import sys

import configparser

sys.path.append('../')

import torchvision
from components.outer_component import DisplayComponent
from components.painter_component import Tiler, BBoxPainter
from dlib import correlation_tracker
from pipeline import Pipeline
from set_stream import *
from yolo import YOLO
import argparse

argparser = argparse.ArgumentParser()

argparser.add_argument('--usbcam', required=False)
argparser.add_argument('--videofile', required=False)
argparser.add_argument('-c', '--config', required=True, help='path to the configuration file')
argparser.add_argument('-t', '--tracking_frames', required=False, help='number of frames without detection')
argparser.add_argument('-w', '--width', required=False, help='video frame width')
argparser.add_argument('-f', '--font', required=False)
argparser.add_argument('--tsize', required=False)
args = vars(argparser.parse_args())

config = configparser.ConfigParser()
config.read(args['config'])
#model = YOLO(clf_spec='vehicles')
model=YOLO()
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
#model_det = get_detection_model('detection', model, sources=readers, classes=VEHICLES_CLASSES)
lines_list = eval(config.get('Lines', 'values'))
lines = []
for line in lines_list:
    lines.append((line['points'][0], line['points'][1], line['color'], line['thickness']))

tracker = get_tracker('tracking', correlation_tracker(), sources=readers, classes=COCO_INSTANCE_CATEGORY_NAMES, lines=lines)
line = [lines[0][0][0], lines[0][0][1], lines[0][1][0], lines[0][1][1]]
counter = get_counter('counter', lines)
bbox_painter = BBoxPainter('bboxer', font_path=args['font'])

tiler = get_tiler('tiler', tiler_size=(2, 2), frame_size=(1440, 2160))

outer = DisplayComponent('display')

pipeline.set_device('cpu')
pipeline.add_all([muxer, model_det, tracker, counter, bbox_painter, tiler, outer])
pipeline.compile()
pipeline.run()
from video import Video
from yolo import YOLO
from sort import CentroidTracker
import time
import argparse
import configparser
from torchvision.datasets import CocoDetection
import torch

from backbone import Backbone
from head import Head


ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=True, help='path to the video')
ap.add_argument('-c', '--config', required=True, help='path to the configuration file')
ap.add_argument('-t', '--tracking_frames', required=False, help='number of frames without detection')
ap.add_argument('-w', '--width', required=False, help='video frame width')
args = vars(ap.parse_args())
config = configparser.ConfigParser()
config.read(args['config'])
backbone = Backbone()
head = Head()
model = YOLO(backbone, head)
path2data = "./train2017"
path2json = "./annotations/instances_train2017.json"
coco_ds = CocoDetection(root=path2data,
                        annFile=path2json)
batch_size = 1
num_workers = 4
#data_loader = torch.utils.data.DataLoader(coco_ds,
#                                          batch_size=batch_size,
#                                          shuffle=False,
#                                          num_workers=num_workers
#                                         )
#model.train(data_loader)
tracker = CentroidTracker(maxDisappeared=0, maxDistance=5000)
lines_list = eval(config.get('Lines', 'values'))
lines = []
for line in lines_list:
    lines.append((line['points'][0], line['points'][1], line['color'], line['thickness']))
video = Video(model, tracker, lines,
              tracking_frames=int(args['tracking_frames']),
              width=int(args['width']) if args.get("width", False) else None)
time.sleep(2.0)
video.set_input(args['video'])
video.process()

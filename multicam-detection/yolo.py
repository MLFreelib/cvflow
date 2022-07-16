import torch
from PIL import Image
from backend.projector.projector import *
from config.config import Config
from camera.calibration.calibration import Calibration
from set_stream import get_config
import Meta
import matplotlib.pyplot as plt
import json
import numpy as np
import dlib
import torchvision.transforms as transforms
from Meta import MetaFrame
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


class YOLO(torch.nn.Module):
    def __init__(self, calib_path=None):
        super().__init__()
        #self.model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
        model_name = 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
        model_weights = 'model_final.pth'
        device = 'cpu'
        detector_config = get_cfg()
        detector_config.merge_from_file(model_zoo.get_config_file(model_name))
        detector_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        detector_config.MODEL.WEIGHTS = model_weights
        detector_config.MODEL.DEVICE = device
        self._predictor = DefaultPredictor(detector_config)
        self.model = DefaultPredictor(detector_config)
        self.boxes = []
        config = get_config()
        self.projector = Projector(Config())
        self.calibrations = []
        self.count = 0
        self.detected = False
        self.trackers = {}
        if calib_path:
            for path_i in calib_path:
                with open(path_i, 'r') as f:
                    points = json.load(f)['calibration']
                screen_points = np.array([])
                world_points = np.array([])
                for world_point, screen_point in zip(points['world_points'], points['screen_points']):
                    screen_points = np.append(screen_points, [screen_point['x'], screen_point['y']])
                    world_points = np.append(world_points, [world_point['x'], world_point['y'], world_point['z']])
                screen_points = screen_points.reshape((6, 2))
                world_points = world_points.reshape((6, 3))
                calibration = Calibration(screen_points, world_points)
                self.calibrations.append(calibration)


    def forward(self, x: Meta):
        if type(x) != Meta.MetaBatch:
            nx = x.cpu().detach().numpy() * 255
            multicam = False
        else:
            multicam = True
            nx = []
            video_id = 0
            frame_id = 1
            video_dict = {}
            frame_id_video_dict = {}
            results = []
            for src_name in x.get_source_names():
                for meta_frame in x.get_meta_frames_by_src_name(src_name):
                    nx.append(meta_frame.get_frame().cpu().detach().numpy() * 255)
                    i = meta_frame.get_frame().cpu().detach().numpy() * 255
                    frame = meta_frame.get_frame().clone()
                    frame *= 255
                    frame = torch.permute(frame, (1, 2, 0))
                    frame = np.array(frame, dtype=np.uint8)
                    if not self.detected:
                        #res = self.model(frame).xyxy[0]
                        output = self._predictor(frame)
                        classes = output["instances"].pred_classes.cpu()
                        bboxes = output["instances"].pred_boxes.tensor.cpu()
                        scores = output["instances"].scores.cpu()
                        res = []
                        for i in range(len(bboxes)):
                            res.append([*(bboxes[i]), classes[i], scores[i]])
                        res = np.array(res)
                        print('RESULT', res)
                        self.trackers[video_id] = []
                        for box in res:
                            tracker = dlib.correlation_tracker()
                            tracker.start_track(frame, dlib.rectangle(*[int(i.item()) for i in box[:4]]))
                            self.trackers[video_id].append(tracker)
                    else:
                        res = []
                        for tracker in self.trackers[video_id]:
                            tracker.update(frame)
                            pos = tracker.get_position()
                            res.append([pos.left(), pos.top(),
                                        pos.right(), pos.bottom(), 1, 0])
                        res = np.array(res)
                    results.append(res)
                    if multicam:
                        box = res[..., :4]
                        bbox = None
                        for bbox in res:
                            if bbox[5] == 39 or bbox[5] == 0:  # TODO: fix for multiple objects
                                break
                        if bbox is not None and len(bbox):
                            d_x = ((bbox[0] + bbox[2]) / 2).item()
                            d_y = ((bbox[1] + bbox[3]) / 2).item()
                            d_c = self.calibrations[video_id].project_2d_to_3d([d_x, d_y], Z=0)
                            d_c_x = d_c[0]
                            d_c_y = d_c[1]
                            video_dict[video_id] = [[d_c_x, d_c_y, 0, 1, 1], ]
                video_id += 1
            if not self.detected:
                self.detected = True
            if self.count % 20 == 0:
                self.detected = False
            frame_id_video_dict[frame_id] = video_dict
        #res = self.model(nx[0]).xyxy
        #print(nx.shape)
        #res = self.model(nx[0]).xyxy[0]
        #self.boxes = res[..., :4]
        out = []
        #res = self.model(nx)
        for i in results:
            #res = self.model(i).xyxy[0]
            #res = torch.tensor(res)
            #if len(res):
            if len(i):
                res = torch.tensor(i)
                out.append({'boxes': res[..., :4],
                            'labels': res[..., 5],
                            'scores': res[..., 4]})
        if multicam:
            projections = self.projector.get_next_projection_batch(frame_id_video_dict)
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            if self.count % 20 != 1:
                plt.plot(projections[0]['points'][0]['x'], projections[0]['points'][0]['y'], colors[self.count % len(colors)] + 'o')
            plt.xlim([-15, 5])
            plt.ylim([-15, 5])
            plt.savefig('map.jpg')
            im = Image.open('map.jpg')
            transform = transforms.Compose([
                transforms.PILToTensor()
            ])
            img_tensor = transform(im)
            meta_img = MetaFrame('plot', img_tensor)
            meta_img.set_frame(img_tensor)
            x.add_meta_frame(meta_img)
            source_names = x.get_source_names()
            source_names.append('plot')
            x.set_source_names(source_names)
        #out = [{'boxes': res[..., :4],
        #        'labels': res[..., 5],
        #        'scores': res[..., 4]}, ]
        self.count += 1
        return out

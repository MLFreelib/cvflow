from typing import List

from Meta import MetaBatch, MetaLabel, MetaBBox
from components.component_base import ComponentBase
import numpy as np
import torch
from models.blocks import DeepSort
import torch
import torch.nn as nn
import torch.nn.functional as F

class TrackerBase(ComponentBase):
    def __init__(self, name: str):
        super().__init__(name)
        self._connected_sources = list()
        self._frame_resolution: tuple = (480, 640)

    def set_confidence(self, conf: float):
        self._confidence = conf

    def set_resolution(self, frame_resolution):
        self._frame_resolution = frame_resolution

    def do(self, data: MetaBatch):
        pass

    def start(self):
        pass

    def set_transforms(self, tensor_transforms: list):
        self.__transforms = tensor_transforms

    def add_source(self, name: str):
        self._connected_sources.append(name)

    def _get_transforms(self):
        return self.__transforms

    def _transform(self, data: torch.Tensor):
        if self._get_transforms():
            for t_transform in self._get_transforms():
                data = t_transform.forward(data)
        return data


class DeepSortTrackerComponent(TrackerBase):
    def __init__(self, name: str, appearance_weights_path):
        super().__init__(name)
        self.tracker = DeepSort(appearance_weights_path)
        self.appearance_weights_path = appearance_weights_path
        self.trackers = {}
        self.ids = []
        self.labels = None
        self.scores = None
        self.boxes = None
        self.counted = 0
        self.frames_count = 0
        self.upd_rate = 20
        self.prev_boxes = []

    def set_labels(self, labels: List[str]):
        self.__label_names = labels

    def do(self, data: MetaBatch) -> MetaBatch:
        if self.frames_count % self.upd_rate == 0:
            print('UPD', self.frames_count)
            self.update(data)
        self.frames_count += 1
        for src_name in data.get_source_names():
            meta_frames = data.get_meta_frames_by_src_name(src_name)
            for meta_frame in meta_frames:
                boxes = []
                frame = meta_frame.get_frame().clone()
                frame *= 255
                frame = torch.permute(frame, (1, 2, 0))
                frame = np.array(frame, dtype=np.uint8)
                print('aaaaaaa')
                bbox_inf = meta_frame.get_bbox_info()
                self.boxes = bbox_inf.get_bbox()
                if (self.frames_count - 1) % self.upd_rate != 0:
                    print('DENORM', frame.shape)
                    self.boxes[:, 0] *= frame.shape[1]
                    self.boxes[:, 1] *= frame.shape[0]
                    self.boxes[:, 2] *= frame.shape[1]
                    self.boxes[:, 3] *= frame.shape[0]
                self.boxes = self.tracker.update(self.boxes, bbox_inf.get_label_info().get_confidence(), frame)
                print(self.boxes)
                '''for box_id in self.trackers[src_name]:
                    tracker = self.trackers[src_name][box_id]
                    tracker['tracker'].update(frame)
                    pos = tracker['tracker'].get_position()
                    box = [pos.left(), pos.top(), pos.right(), pos.bottom()]
                    boxes.append(box)'''
                print('ddddddd')
                self.boxes = torch.tensor(self.boxes[:len(self.labels)]).type(torch.float)
                shape = meta_frame.get_frame().shape
                print(self.boxes.dtype, shape[2])
                #self.boxes[:, (0, 2)] = (self.boxes[:, (0, 2)] / shape[2]).type(torch.int64) 
                #self.boxes[:, (1, 3)] = (self.boxes[:, (1, 3)] / shape[1]).type(torch.int64)
                print('bbbbbb')
                labels = self.labels
                scores = self.scores
                meta_labels = MetaLabel(labels, scores)
                if self.boxes.shape == 1:
                    self.update(data)
                print('CCC_BOXES', self.boxes)
                self.boxes[:, 0] /= frame.shape[1]
                self.boxes[:, 1] /= frame.shape[0]
                self.boxes[:, 2] /= frame.shape[1]
                self.boxes[:, 3] /= frame.shape[0]
                meta_box = MetaBBox(self.boxes, meta_labels)
                meta_frame.set_bbox_info(meta_box)
                print('ccccccccc', meta_frame.get_bbox_info().get_bbox())
                self.ids = [i for i in range(len(self.boxes))]
                meta_frame.get_bbox_info().get_label_info().set_object_id(self.ids)
                meta_bbox = meta_frame.get_bbox_info()
                meta_label = meta_bbox.get_label_info()
                data.get_meta_frames_by_src_name(src_name)[0].set_label_info(meta_labels)
                print('IDS', self.ids)
                meta_label.set_object_id(self.ids)
                print('RETURN data')
        return data

    def update(self, data: MetaBatch):
        print('UPD COUNT', self.frames_count)
        box_id = 0
        self.old_trackers = self.trackers
        self.trackers = {}
        for src_name in data.get_source_names():
            self.ids = []
            if src_name not in self.trackers:
                self.trackers[src_name] = {}
            for meta_frame in data.get_meta_frames_by_src_name(src_name):
                if meta_frame.get_bbox_info() is None:
                    boxes = torch.tensor([[0, 0, 0, 0]])
                    labels = ['', ]
                    scores = [0, ]
                    meta_labels = MetaLabel(labels, scores)
                    meta_box = MetaBBox(boxes, meta_labels)
                    meta_frame.set_bbox_info(meta_box)
                else:
                    boxes = meta_frame.get_bbox_info().get_bbox()
                    try:
                        self.labels = meta_frame.get_bbox_info().get_label_info().get_labels()
                        self.scores = meta_frame.get_bbox_info().get_label_info().get_confidence()
                    except AttributeError:
                        self.labels = ['N/A'] * len(self.boxes)
                        self.scores = [0] * len(self.boxes)
                self.boxes = boxes
                shape = meta_frame.get_frame().shape
                self.boxes[:, (0, 2)] = self.boxes[:, (0, 2)] * shape[2]
                self.boxes[:, (1, 3)] = self.boxes[:, (1, 3)] * shape[1]
                frame = meta_frame.get_frame().clone()
                frame *= 255
                frame = torch.permute(frame, (1, 2, 0))
                frame = np.array(frame, dtype=np.uint8)
                for box in self.boxes:
                    box_id += 1
                    existing_box_id = self.find_box(box, src_name, 0.6)
                    checked = False
                    if existing_box_id:
                        self.ids.append(existing_box_id)
                        try:
                            checked = self.old_trackers[src_name][existing_box_id]['checked']
                        except KeyError:
                            checked = False
                    else:
                        self.ids.append(box_id)
                    #rect = dlib.rectangle(*box)
                    #tracker = correlation_tracker()
                    #tracker.start_track(frame, rect)
                    self.trackers[src_name][box_id] = {'tracker': self.tracker, 'box': box, 'checked': checked}
                meta_frame.get_bbox_info().get_label_info().set_object_id(self.ids)

    def find_box(self, box, src_name, threshold):
        try:
            trackers = self.old_trackers[src_name]
        except KeyError:
            trackers = self.trackers[src_name]
        for box_id in trackers:
            cur_box = trackers[box_id]['box']
            intersection = [max(box[0], cur_box[0]),
                            max(box[1], cur_box[1]),
                            min(box[2], cur_box[2]),
                            min(box[3], cur_box[3])]
            intersection_w = intersection[2] - intersection[0]
            intersection_h = intersection[3] - intersection[1]
            box_w = box[2] - box[0]
            box_h = box[3] - box[1]
            if intersection_w <= 0 or intersection_h <= 0:
                continue
            box_area = box_w * box_h
            intersection_area = (intersection_w * intersection_h / box_area) ** 0.5
            if intersection_area >= threshold:
                return box_id
        return None
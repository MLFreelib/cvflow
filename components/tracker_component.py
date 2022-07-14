from typing import List

from Meta import MetaBatch, MetaLabel, MetaBBox
from components.component_base import ComponentBase
#from tracker import Tracker
from dlib import correlation_tracker
import numpy as np
import dlib
import torch
import cv2


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
        # self._inference.to(device=self.get_device())

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


class CorrelationBasedTrackerComponent(TrackerBase):
    def __init__(self, name: str):
        super().__init__(name)
        #self.tracker = Tracker()
        self.tracker = correlation_tracker()
        self.trackers = []
        self.ids = []
        self.boxes = None
        self.counted = 0

    def set_labels(self, labels: List[str]):
        self.__label_names = labels

    def do(self, data: MetaBatch) -> MetaBatch:
        print('DO')
        for src_name in data.get_source_names():
            meta_frames = data.get_meta_frames_by_src_name(src_name)
            shape = data.get_frames_by_src_name(src_name).shape
            for meta_frame in meta_frames:
                boxes = []
                frame = meta_frame.get_frame().clone()
                frame *= 255
                frame = torch.permute(frame, (1, 2, 0))
                frame = np.array(frame, dtype=np.uint8)
                for tracker in self.trackers:
                    tracker['tracker'].update(frame)
                    pos = tracker['tracker'].get_position()
                    box = [pos.left(), pos.top(), pos.right(), pos.bottom()]
                    boxes.append(box)
                self.boxes = torch.tensor(boxes)
                labels = ['object'] * len(self.ids)
                scores = [1] * len(self.ids)
                meta_labels = MetaLabel(labels, scores)
                if self.boxes.shape == 1:
                    self.update(data)
                #print('BOXES SHAPE-------', self.boxes.shape)
                meta_box = MetaBBox(self.boxes, meta_labels)
                meta_frame.set_bbox_info(meta_box)
                meta_frame.get_bbox_info().get_label_info().set_object_id(self.ids)
                #print('GET BBOX', meta_frame.get_bbox_info().get_bbox())
                meta_bbox = meta_frame.get_bbox_info()
                meta_label = meta_bbox.get_label_info()
                data.get_meta_frames_by_src_name(src_name)[0].set_label_info(meta_labels)
                #('GET BBOX2', data.get_meta_frames_by_src_name(src_name)[0].get_bbox_info().get_bbox())
                meta_label.set_object_id(self.ids)
                #print('IDS', self.ids)
        return data

    def update(self, data: MetaBatch):
        print('UPDATE')
        box_id = 0
        self.trackers = []
        self.ids = []
        for src_name in data.get_source_names():
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
                self.boxes = boxes
                shape = meta_frame.get_frame().shape
                self.boxes[:, (0, 2)] = self.boxes[:, (0, 2)] * shape[2]
                self.boxes[:, (1, 3)] = self.boxes[:, (1, 3)] * shape[1]

                for box in boxes:
                    box_id += 1
                    self.ids.append(box_id)
                    rect = dlib.rectangle(*box)
                    frame = meta_frame.get_frame().clone()
                    frame *= 255
                    frame = torch.permute(frame, (1, 2, 0))
                    frame = np.array(frame, dtype=np.uint8)
                    print(type(frame), type(rect))
                    tracker = correlation_tracker()
                    tracker.start_track(frame, rect)
                    print('TRCAKEER', tracker)
                    checked = False
                    self.trackers.append({'tracker': tracker, 'checked': checked})
                meta_frame.get_bbox_info().get_label_info().set_object_id(self.ids)


class ManualROICorrelationBasedTracker(TrackerBase):
    r""" Gets bboxes, tracks objects and calculate distance between. """

    def __init__(self, name: str, boxes):
        super().__init__(name)
        # self.tracker = Tracker()
        self.tracker = correlation_tracker()
        self.trackers = []
        self.ids = []
        self.boxes = boxes
        self.start_bb_not_set = True

    def set_labels(self, labels: List[str]):
        self.__label_names = labels

    def __bbox_normalize(self, bboxes: torch.tensor, shape: torch.tensor):
        r""" Normalization of bounding box values in the range from 0 to 1.
            :param bboxes: torch.tensor
            :param shape: torch.tensor - image resolution.
            :return:
        """
        bboxes = bboxes.float()
        bboxes[:, (0, 2)] = bboxes[:, (0, 2)].div(shape[3])
        bboxes[:, (1, 3)] = bboxes[:, (1, 3)].div(shape[2])
        return bboxes

    def do(self, data: MetaBatch) -> MetaBatch:
        print('DO')
        if self.start_bb_not_set:
            self.set_trackers(data)
        for src_name in data.get_source_names():
            meta_frames = data.get_meta_frames_by_src_name(src_name)
            shape = data.get_frames_by_src_name(src_name).shape
            for meta_frame in meta_frames:
                boxes = []
                frame = meta_frame.get_frame().clone()
                frame *= 255
                frame = torch.permute(frame, (1, 2, 0))
                frame = np.array(frame, dtype=np.uint8)
                for tracker in self.trackers:
                    tracker['tracker'].update(frame)
                    pos = tracker['tracker'].get_position()
                    box = [pos.left(), pos.top(), pos.right(), pos.bottom()]
                    boxes.append(box)
                boxes = self.__bbox_normalize(torch.tensor(boxes), shape)
                labels = ['object'] * len(self.ids)
                scores = [1] * len(self.ids)
                meta_labels = MetaLabel(labels, scores)
                # print('BOXES SHAPE-------', self.boxes.shape)
                meta_box = MetaBBox(boxes, meta_labels)
                meta_frame.set_bbox_info(meta_box)
                meta_frame.get_bbox_info().get_label_info().set_object_id(self.ids)
                # print('GET BBOX', meta_frame.get_bbox_info().get_bbox())
                meta_bbox = meta_frame.get_bbox_info()
                meta_label = meta_bbox.get_label_info()
                data.get_meta_frames_by_src_name(src_name)[0].set_label_info(meta_labels)
                # ('GET BBOX2', data.get_meta_frames_by_src_name(src_name)[0].get_bbox_info().get_bbox())
                meta_label.set_object_id(self.ids)
                # print('IDS', self.ids)
        return data

    def set_trackers(self, data):
        print('setting trackers')
        box_id = 0
        self.trackers = []
        self.ids = []

        first_frame = data.get_meta_frames_by_src_name(data.get_source_names()[0])[0]
        shape = data.get_frames_by_src_name(data.get_source_names()[0]).shape
        for box in self.boxes:
            box_id += 1
            self.ids.append(box_id)
            rect = dlib.rectangle(*box)
            frame = first_frame.get_frame().clone()
            frame *= 255
            frame = torch.permute(frame, (1, 2, 0))
            frame = np.array(frame, dtype=np.uint8)
            tracker = correlation_tracker()
            tracker.start_track(frame, rect)
            print('TRACKER', tracker)
            checked = False
            self.trackers.append({'tracker': tracker, 'checked': checked})
        boxes = self.__bbox_normalize(torch.tensor(self.boxes), shape)

        labels = ['object'] * len(self.ids)
        scores = [1] * len(self.ids)
        meta_labels = MetaLabel(labels, scores)
        meta_box = MetaBBox(boxes, meta_labels)
        first_frame.set_bbox_info(meta_box)
        first_frame.get_bbox_info().get_label_info().set_object_id(self.ids)
        self.start_bb_not_set = False
from typing import List

from Meta import MetaBatch, MetaLabel, MetaBBox, MetaName
from components.component_base import ComponentBase
import cv2
import numpy as np
import torch

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


class ManualROICorrelationBasedTracker(TrackerBase):
    r""" Gets bboxes, tracks objects and calculate distance between. """

    def __init__(self, name: str, boxes, tracker_type: str = 'KCF'):
        super().__init__(name)
        self.trackers = []
        self.ids = []
        self.boxes = boxes
        self.start_bb_not_set = True
        self.tracker_type = tracker_type

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

    def __get_tracker(self, tracker_type):

        if tracker_type == 'BOOSTING':
            return cv2.legacy.TrackerBoosting_create()
        if tracker_type == 'MIL':
            return cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            return cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            return cv2.legacy.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            return cv2.legacy.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            return cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            return cv2.legacy.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            return cv2.TrackerCSRT_create()

    def do(self, data: MetaBatch) -> MetaBatch:

        if self.start_bb_not_set:
            self.set_trackers(data)

        for src_name in data.get_source_names():
            meta_frames = data.get_meta_frames_by_src_name(src_name)
            shape = data.get_frames_by_src_name(src_name).shape
            for meta_frame in meta_frames:
                boxes = []
                frame = meta_frame.get_frame().clone().cpu()
                frame *= 255
                frame = torch.permute(frame, (1, 2, 0))
                frame = np.array(frame, dtype=np.uint8)
                for tracker in self.trackers:
                    success, box = tracker.update(frame)
                    if success:
                        boxes.append([box[0], box[1], box[0] + box[2], box[1] + box[3]])

                boxes = self.__bbox_normalize(torch.tensor(boxes), shape)
                labels = ['object'] * len(self.ids)
                scores = [1] * len(self.ids)
                meta_labels = MetaLabel(labels, scores)
                meta_box = MetaBBox(boxes, meta_labels)

                meta_frame.add_meta(MetaName.META_BBOX.value, meta_box)
                self.ids = [i for i in range(len(self.boxes))]
                meta_frame.get_meta_info(MetaName.META_BBOX.value).get_label_info().set_object_id(self.ids)
                meta_bbox = meta_frame.get_meta_info(MetaName.META_BBOX.value)
                meta_label = meta_bbox.get_label_info()
                data.get_meta_frames_by_src_name(src_name)[0].add_meta(MetaName.META_LABEL.value, meta_labels)
                meta_label.set_object_id(self.ids)

        return data

    def set_trackers(self, data):
        box_id = 0
        self.trackers = []
        self.ids = []

        first_frame = data.get_meta_frames_by_src_name(data.get_source_names()[0])[0]
        shape = data.get_frames_by_src_name(data.get_source_names()[0]).shape
        for box in self.boxes:
            box_id += 1
            self.ids.append(box_id)
            rect = (box[0], box[1], box[2] - box[0], box[3] - box[1])
            frame = first_frame.get_frame().clone().cpu()
            frame *= 255
            frame = torch.permute(frame, (1, 2, 0))
            frame = np.array(frame, dtype=np.uint8)
            tracker = self.__get_tracker(self.tracker_type)
            tracker.init(frame, rect)
            self.trackers.append(tracker)
        boxes = self.__bbox_normalize(torch.tensor(self.boxes), shape)

        labels = ['object'] * len(self.ids)
        scores = [1] * len(self.ids)
        meta_labels = MetaLabel(labels, scores)
        meta_box = MetaBBox(boxes, meta_labels)
        first_frame.add_meta(MetaName.META_BBOX.value, meta_box)
        first_frame.get_meta_info(MetaName.META_BBOX.value).get_label_info().set_object_id(self.ids)
        self.start_bb_not_set = False

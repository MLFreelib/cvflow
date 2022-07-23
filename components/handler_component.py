from typing import List, Iterable

import cv2
import numpy as np
import pandas as pd
import torch

from Meta import MetaBatch, MetaFrame, MetaLabel, MetaBBox
from components.component_base import ComponentBase


class Filter(ComponentBase):
    r""" Filters labels in metadata, leaving only those that are passed to the component."""

    def __init__(self, name: str, labels: List[str]):
        r"""
        :param name: str
                    name of component
        :param labels: List[str]
                    the names of the labels that need to be left.
        """
        super().__init__(name)
        self.__mask_labels = labels

    def start(self):
        r""" Checking the type of labels being transmitted. """
        if not isinstance(self.__mask_labels, list):
            raise TypeError(f"Expected List[str], actual {type(self.__mask_labels)}")

    def do(self, data: MetaBatch) -> MetaBatch:
        r""" Filters labels.
            :param data: MetaBatch
                        metadata about batch.
        """

        for source in data.get_source_names():
            for meta_frame in data.get_meta_frames_by_src_name(source):
                self.__filter_labels(meta_frame)
                self.__filter_bboxes(meta_frame)
                self.__filter_masks(meta_frame)

        return data

    def __filter_labels(self, meta_frame: MetaFrame):
        r""" Filters labels for classification models.
            :param: meta_frame: MetaFrame
                        metadata about frame.
        """
        meta_labels = meta_frame.get_labels_info()
        if meta_labels is not None:
            labels = meta_labels.get_labels()
            sr_labels = pd.Series(False, labels)
            indexes = list(set(labels) & set(self.__mask_labels))
            sr_labels[indexes] = True
            confs = meta_labels.get_confidence().clone()
            confs = confs[list(sr_labels.values)]
            meta_frame.set_label_info(MetaLabel(list(sr_labels.index[sr_labels.values]), torch.unsqueeze(confs, dim=0)))

    def __filter_bboxes(self, meta_frame: MetaFrame):
        r""" Filters labels and bboxes for detection models.
            :param: meta_frame: MetaFrame
                        metadata about frame.
        """
        meta_bboxes = meta_frame.get_bbox_info()
        if meta_bboxes is not None:
            meta_labels = meta_bboxes.get_label_info()
            labels = meta_labels.get_labels()
            sr_labels = pd.Series(False, labels)
            indexes = list(set(labels) & set(self.__mask_labels))
            sr_labels[indexes] = True
            labels_id = list(sr_labels.values)
            bboxes = meta_bboxes.get_bbox()[labels_id]
            new_meta_label = MetaLabel(list(sr_labels[sr_labels].index), meta_labels.get_confidence()[labels_id])
            ids = meta_labels.get_object_ids()
            if len(ids) == len(labels_id):
                new_meta_label.set_object_id([ids[i] for i, e in enumerate(labels_id) if e == True])
            new_meta_bbox = MetaBBox(bboxes, new_meta_label)
            meta_frame.set_bbox_info(new_meta_bbox)

    def __filter_masks(self, meta_frame: MetaFrame):
        r""" Filters labels and masks for segmentation models.
            :param: meta_frame: MetaFrame
                        metadata about frame.
        """
        meta_masks = meta_frame.get_mask_info()
        if meta_masks is not None:
            meta_labels = meta_masks.get_label_info()
            labels = meta_labels.get_labels()
            sr_labels = pd.Series(False, labels)
            indexes = list(set(labels) & set(self.__mask_labels))
            sr_labels[indexes] = True
            masks = meta_masks.get_mask()
            masks[:, ~sr_labels.values, :, :] = 0
            meta_masks.set_mask(masks)


class Counter(ComponentBase):
    r""" Draws a line and counts objects by ID that intersect this line. """

    def __init__(self, name: str, lines):
        r"""
            :param name: str
                    name of component.
            :param line: List[int]
                    the line along which the objects will be counted. Format: [x_min, y_min, x_max, y_max]
        """
        super().__init__(name)
        self.__lines = lines
        self.__label_count = dict()
        self.__checked_ids = dict()

    def do(self, data: MetaBatch) -> MetaBatch:
        r""" Counts objects. """
        for src_name in data.get_source_names():
            meta_frames = data.get_meta_frames_by_src_name(src_name)
            for meta_frame in meta_frames:
                frame = meta_frame.get_frame()
                frame = self.__draw_line(frame)
                meta_frame.set_frame(frame)
                if meta_frame.get_bbox_info() is not None:
                    self.__update(meta_frame.get_bbox_info(),
                                  meta_frame.get_frame().detach().cpu().numpy().shape, src_name)
                if src_name in list(self.__label_count.keys()):
                    meta_frame.add_meta('counter', self.__label_count[src_name])
        return data

    def __update(self, meta_bbox: MetaBBox, shape: Iterable[int], source: str):
        r""" Updates the current number of counted objects.
            :param meta_bbox: MetaBBox
                            metadata about the bounding boxes for the frame.
            :param shape: Iterable[int]
                            shape of frame.
            :param source: str
                            the source from which the frame was received.
        """
        if source not in list(self.__checked_ids.keys()):
            self.__checked_ids[source] = dict()
            self.__label_count[source] = {'labels': dict(), 'ids': dict()}

        checked_ids = list()
        bboxes = meta_bbox.get_bbox()
        label_info = meta_bbox.get_label_info()
        object_ids = [i for i in range(len(label_info.get_labels()))]
        #object_ids = label_info.get_object_ids()
        labels = label_info.get_labels()
        for i in range(bboxes.shape[0]):
            for line in self.__lines:
                is_intersect = self.__check_intersect(bboxes[i].clone(), line, shape)
                if is_intersect:
                    print('INTERSECT')
                    print('\a')
                    object_id = object_ids[i]
                    label = labels[i]
                    if label not in list(self.__label_count[source]['labels'].keys()):
                        self.__label_count[source]['labels'][label] = 0
                    if object_id not in list(self.__checked_ids[source].keys()):
                        self.__label_count[source]['ids'][object_id] = 1
                        self.__label_count[source]['labels'][label] += 1
                    elif not self.__checked_ids[source][object_id]:
                        self.__label_count[source]['ids'][object_id] += 1
                        self.__label_count[source]['labels'][label] += 1
                    self.__checked_ids[source][object_id] = True
                    checked_ids.append(object_id)

        for object_id in list(self.__checked_ids[source].keys()):
            if object_id not in checked_ids:
                self.__checked_ids[source][object_id] = False

    def __draw_line(self, frame: torch.Tensor):
        r""" Draws a line along which objects are counted.

            :param frame: torch.Tensor
                        the frame on which the line will be drawn.
        """
        frame = frame.detach().cpu()
        frame = frame.permute(1, 2, 0).numpy()
        frame = np.ascontiguousarray(frame)
        for line in self.__lines:
            print(line)
            frame = cv2.line(frame, line[0], line[1], color=line[2], thickness=line[3])
        return torch.tensor(frame, device=self.get_device()).permute(2, 0, 1)

    def __check_intersect(self, bbox: torch.Tensor, line, shape: Iterable[int]) -> bool:
        r""" Checks whether the object crosses the line.
            :param bbox: torch.Tensor
                        bounding box.
            :param shape: tuple
                        shape of frame.
        """
        cv_shape = (*shape[1:], shape[0])
        self.__bbox_denormalize(torch.unsqueeze(bbox, dim=0), shape)
        np_bbox = bbox.detach().cpu().numpy().astype(int)
        check_line = cv2.line(np.zeros(cv_shape), line[0], line[1], thickness=line[3], color=(255, 255, 255))
        # check_bbox = cv2.rectangle(np.zeros(cv_shape), np_bbox[:2], np_bbox[2:], color=(255, 255, 255), thickness=-1)
        check_bbox = cv2.line(np.zeros(cv_shape), (np_bbox[0], np_bbox[3]), (np_bbox[2], np_bbox[3]), color=(255, 255, 255), thickness=5)
        dif = check_bbox - check_line
        dif[dif < 0] = 0

        if np.sum(check_bbox) != np.sum(dif):
            return True
        else:
            return False

    def __bbox_denormalize(self, bboxes: torch.tensor, shape: torch.tensor):
        r""" Gets coordinates for bounding boxes.
            :param bboxes: torch.tensor
                        bounding boxes. shape: [N, 4]
            :param shape: torch.tensor
                        frame resolution
        """
        bboxes[:, (0, 2)] = bboxes[:, (0, 2)].mul(shape[2])
        bboxes[:, (1, 3)] = bboxes[:, (1, 3)].mul(shape[1])

    def stop(self):
        print(self.__label_count)
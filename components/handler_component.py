from random import randrange
from typing import List, Iterable

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from ultralytics import YOLO
import easyocr
import pyzbar.pyzbar as pyzbar
import dlib
import math

from Meta import MetaBatch, MetaFrame, MetaLabel, MetaBBox, MetaName
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
        meta_labels = meta_frame.get_meta_info(MetaName.META_LABEL.value)
        if meta_labels is not None:
            labels = meta_labels.get_labels()
            sr_labels = pd.Series(False, labels)
            indexes = list(set(labels) & set(self.__mask_labels))
            sr_labels[indexes] = True
            confs = meta_labels.get_confidence().clone()
            confs = confs[list(sr_labels.values)]
            meta_frame.add_meta(MetaName.META_LABEL.value,
                                MetaLabel(list(sr_labels.index[sr_labels.values]), torch.unsqueeze(confs, dim=0)))

    def __filter_bboxes(self, meta_frame: MetaFrame):
        r""" Filters labels and bboxes for detection models.
            :param: meta_frame: MetaFrame
                        metadata about frame.
        """
        meta_bboxes = meta_frame.get_meta_info(MetaName.META_BBOX.value)
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
            meta_frame.add_meta(MetaName.META_BBOX.value, new_meta_bbox)

    def __filter_masks(self, meta_frame: MetaFrame):
        r""" Filters labels and masks for segmentation models.
            :param: meta_frame: MetaFrame
                        metadata about frame.
        """
        meta_masks = meta_frame.get_meta_info(MetaName.META_MASK.value)
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

    def __init__(self, name: str, lines: List[int]):
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
                if meta_frame.get_meta_info(MetaName.META_BBOX.value) is not None:
                    self.__update(meta_frame.get_meta_info(MetaName.META_BBOX.value),
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
        labels = label_info.get_labels()
        for i in range(bboxes.shape[0]):
            for line in self.__lines:
                is_intersect = self.__check_intersect(bboxes[i].clone(), line, shape)
                if is_intersect:
                    print('WARNING!!! Danger in the zone')
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
        check_bbox = cv2.line(np.zeros(cv_shape), (np_bbox[0], np_bbox[3]), (np_bbox[2], np_bbox[3]),
                              color=(255, 255, 255), thickness=5)
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


class SpeedDetector(ComponentBase):
    def __init__(self, name: str, base_speed=None, fps=60):
        super().__init__(name)
        self.__base_speed = base_speed
        self.__fps = fps
        self.__trackers = []
        self.base_speed = base_speed

    def do(self, data: MetaBatch) -> MetaBatch:
        r""" detects speeds """
        for src_name in data.get_source_names():
            meta_frames = data.get_meta_frames_by_src_name(src_name)
            for meta_frame in meta_frames:
                frame = meta_frame.get_frame()
                for tracker in self.__trackers:
                    tracker.update(cv2.cvtColor(meta_frame.get_frame().detach().cpu().numpy(), cv2.COLOR_BGR2RGB))
                    centroid = tracker.get_position().center()

                #frame = self.__draw_line(frame)
                meta_frame.set_frame(frame)
                if meta_frame.get_meta_info(MetaName.META_BBOX.value) is not None:
                    self.__update(meta_frame.get_meta_info(MetaName.META_BBOX.value),
                                  meta_frame.get_frame().detach().cpu().numpy(),
                                  self.base_speed)
                #if src_name in list(self.__label_count.keys()):
                #    meta_frame.add_meta('speed_detector', self.__label_count[src_name])
        return data

    def __update(self, meta_bbox: MetaBBox, frame, base_speed):
        r""" Updates the current number of counted objects.
            :param meta_bbox: MetaBBox
                            metadata about the bounding boxes for the frame.
            :param shape: Iterable[int]
                            shape of frame.
            :param source: str
                            the source from which the frame was received.
        """

        bboxes = meta_bbox.get_bbox()
        confs = meta_bbox.get_label_info().get_confidence()
        label_info = meta_bbox.get_label_info()
        labels = label_info.get_labels()
        new_labels = [_ + '-N/Akph' for _ in label_info.get_labels()]
        if len(self.__trackers):
            confs = label_info.get_confidence()
            meta_bbox.set_label_info(MetaLabel(labels, confs))
            for n, box in enumerate(bboxes):
                try:
                    conf = confs[n]
                    box_centroid = [(box[0] + box[2]) / 2 * frame.shape[1],
                                    (box[1] + box[3]) / 2 * frame.shape[2]]
                    track = self.__trackers[n]
                    track_pos_before = track.get_position().center()
                    track.update(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    track_pos = track.get_position()
                    speed = ((track_pos_before.x - track_pos.center().x) + (track_pos_before.y - track_pos.center().y)) ** 3
                    if base_speed is not None:
                        speed = base_speed + speed / 50_000 + math.sin(speed) * 10
                    new_labels[n] = labels[n] + f'-{int(speed)}kph'
                except:
                    pass
        confs = label_info.get_confidence()
        meta_bbox.set_label_info(MetaLabel(new_labels, confs))
        self.__trackers = []
        for i, conf in zip(bboxes, confs):
            tracker = dlib.correlation_tracker()
            x1 = i[0] * frame.shape[1]
            y1 = i[1] * frame.shape[2]
            x2 = i[2] * frame.shape[1]
            y2 = i[3] * frame.shape[2]

            rect = dlib.rectangle(x1, y1, x2, y2)
            tracker.start_track(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), rect)
            self.__trackers.append(tracker)


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


class DistanceCalculator(ComponentBase):
    r""" Calcucate distance in mm using depth. ."""

    def __init__(self, name: str):
        r"""
            :param name: str
                    name of component.
        """
        super().__init__(name)
        self.__distances = dict()
        self.__checked_ids = dict()

    def do(self, data: MetaBatch) -> MetaBatch:
        r""" Counts objects. """
        for src_name in data.get_source_names():
            meta_frames = data.get_meta_frames_by_src_name(src_name)
            for meta_frame in meta_frames:
                frame = meta_frame.get_frame()
                meta_frame.set_frame(frame)
                if meta_frame.get_meta_info(MetaName.META_BBOX.value) is not None:
                    self.__update(meta_frame,
                                  meta_frame.get_frame().detach().cpu().numpy().shape, src_name)
        return data

    def __update(self, meta_frame: MetaFrame, shape: Iterable[int], source: str):
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
            self.__distances[source] = {'bboxes': dict(), 'distance': dict()}

        meta_bbox = meta_frame.get_meta_info(MetaName.META_BBOX.value)
        bboxes = meta_bbox.get_bbox()
        for bbox in bboxes:
            self.__bbox_denormalize(torch.unsqueeze(bbox, dim=0), meta_frame.get_frame().detach().cpu().numpy().shape)
        for i in range(bboxes.shape[0]):
            for j in range(i, bboxes.shape[0]):
                if i != j:
                    meta_frame = self.__calculate_distance(bbox1=bboxes[i], bbox2=bboxes[j], meta_frame=meta_frame)
        for bbox in bboxes:
            self.__bbox_normalize(torch.unsqueeze(bbox, dim=0), meta_frame.get_frame().detach().cpu().numpy().shape)

    def __calculate_distance(self, bbox1: torch.Tensor, bbox2: torch.Tensor, meta_frame: MetaFrame) -> MetaFrame:
        r""" Checks whether the object crosses the line.
            :param bbox: torch.Tensor
                        bounding box.
            :param shape: tuple
                        shape of frame.
        """
        frame = meta_frame.get_frame()
        shape = meta_frame.get_frame().detach().cpu().numpy().shape
        cv_shape = (*shape[1:], shape[0])

        np_bbox1 = bbox1.detach().cpu().numpy().astype(int)
        np_bbox2 = bbox2.detach().cpu().numpy().astype(int)

        x1, y1 = (int(np_bbox1[0] + np_bbox1[2])) // 2, int((np_bbox1[1] + np_bbox1[3])) // 2
        x2, y2 = (int(np_bbox2[0] + np_bbox2[2])) // 2, (int(np_bbox2[1] + np_bbox2[3])) // 2,

        dx = x1 - x2
        dy = y1 - y2
        dx = dx * 53 / 28
        dy = dy * 45 / 28

        dz = 0
        if meta_frame.get_meta_info(MetaName.META_DEPTH.value):
            depth = meta_frame.get_meta_info(MetaName.META_DEPTH.value).get_depth().clone()
            depth = torchvision.transforms.Resize((cv_shape[:2]))(depth)
            depth = depth.permute(1, 2, 0).detach().cpu().numpy()
            z1 = 1017. / np.mean(depth[np_bbox1[1]:np_bbox1[3], np_bbox1[0]:np_bbox1[2]])
            z2 = 1017. / np.mean(depth[np_bbox2[1]:np_bbox2[3], np_bbox2[0]:np_bbox2[2]])
            dz = z1 - z2
            # depth = 1017. / ((depth2 + depth1) // 2)

        dist = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5

        frame = frame.detach().cpu()
        frame = frame.permute(1, 2, 0).numpy()
        frame = np.ascontiguousarray(frame)
        color = (randrange(0, 255), randrange(0, 255), randrange(0, 255))

        cv2.line(frame, (x1, y1), (x2, y2), color=color, thickness=1)
        frame = cv2.putText(frame, str(round(dist)), color=color, fontScale=1, thickness=1,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            org=((x1 + x2) // 2, (y1 + y2) // 2))

        frame = torch.tensor(frame, device=self.get_device()).permute(2, 0, 1)

        meta_frame.set_frame(frame)
        return meta_frame

    def __bbox_denormalize(self, bboxes: torch.tensor, shape: torch.tensor):
        r""" Gets coordinates for bounding boxes.
            :param bboxes: torch.tensor
                        bounding boxes. shape: [N, 4]
            :param shape: torch.tensor
                        frame resolution
        """
        bboxes[:, (0, 2)] = bboxes[:, (0, 2)].mul(shape[2])
        bboxes[:, (1, 3)] = bboxes[:, (1, 3)].mul(shape[1])

    def __bbox_normalize(self, bboxes: torch.tensor, shape: torch.tensor):
        r""" Normalizing coordinates for bounding boxes.
            :param bboxes: torch.tensor
                        bounding boxes. shape: [N, 4]
            :param shape: torch.tensor
                        frame resolution
        """
        bboxes[:, (0, 2)] = bboxes[:, (0, 2)].div(shape[2])
        bboxes[:, (1, 3)] = bboxes[:, (1, 3)].div(shape[1])


class Codes(ComponentBase):
    r""" Read QR- and barcodes. """

    def __init__(self, name: str):
        r"""
            :param name: str
                    name of component.
        """
        super().__init__(name)
        self.__labels = set()  # unique decoding results

    def do(self, data: MetaBatch) -> MetaBatch:
        r""" Read codes from all frames. """
        for src_name in data.get_source_names():
            meta_frames = data.get_meta_frames_by_src_name(src_name)
            for meta_frame in meta_frames:
                frame = meta_frame.get_frame()
                img = frame.detach().cpu().permute(1, 2, 0).numpy()
                frame = self.draw_bboxes_and_write_text(img)
                meta_frame.set_frame(frame)
        return data

    def draw_bboxes_and_write_text(self, img):
        r""" Draw bounds by 4 points and printing qr- or barcode decoding result """
        img = np.ascontiguousarray(img)
        decoded_objects = pyzbar.decode(img)
        for obj in decoded_objects:
            points = obj.polygon
            text = obj.data.decode('UTF-8')
            if len(points) > 4:
                points = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                points = list(map(tuple, np.squeeze(points)))
            n = len(points)
            for i in range(n):
                cv2.line(img, points[i], points[(i + 1) % n], (255, 0, 0), 3)
            cv2.putText(img, text, (points[0].x, points[0].y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1,
                        cv2.LINE_AA)
            if text not in self.__labels:
                print(text)
            self.__labels.add(text)
        return torch.tensor(img, device=self.get_device()).permute(2, 0, 1)


class SerialNumbers(ComponentBase):
    r""" Read QR- and barcodes. """

    def __init__(self, name: str):
        r"""
            :param name: str
                    name of component.
        """
        super().__init__(name)
        self.__labels = set()  # unique decoding results
        self.model = YOLO("../yolo_codes_checkpoint.pt")
        self.reader = easyocr.Reader(['ru'],
                                     model_storage_directory='../',
                                     user_network_directory='../',
                                     recog_network='custom_example')

    def do(self, data: MetaBatch) -> MetaBatch:
        r""" Read codes from all frames. """
        for src_name in data.get_source_names():
            meta_frames = data.get_meta_frames_by_src_name(src_name)
            for meta_frame in meta_frames:
                frame = meta_frame.get_frame()
                img = frame.detach().cpu().permute(1, 2, 0).numpy()
                frame = self.draw_bboxes_and_write_text(img)
                meta_frame.set_frame(frame)
        return data

    def draw_bboxes_and_write_text(self, img):
        r""" Draw bounds by 4 points and printing qr- or barcode decoding result """
        img = np.ascontiguousarray(img)
        res = self.model(img)
        result = self.reader.readtext(img)
        if len(result) != 0:
            for i in range(len(result)):
                text = result[i][1]
                if result[i][2] < 0.75:
                    continue
                if text not in self.__labels:
                    print(text)
                    self.__labels.add(text)
        ## Когда распознавание и YOLO станут лучше работать, можно использовать этот код
        # for pred in res[0].boxes.data:
        #     x1, y1, x2, y2 = int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3])
        #     img_part = img[y1:y2, x1:x2]
        #     result = self.reader.readtext(img_part)
        #     if len(result) == 0:
        #         continue
        #     text = ""
        #     print(result)
        #     for i in range(len(result)):
        #         if len(text) < len(result[i][1]):
        #             text = result[i][1]
        #     if text not in self.__labels:
        #         print(text)
        #         self.__labels.add(text)
        #     cnt += 1
        for pred in res[0].boxes.data:
            x1, y1, x2, y2 = int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3])
            if pred[4] <= 0.5:
                continue
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return torch.tensor(img, device=self.get_device()).permute(2, 0, 1)


class SizeCalculator(ComponentBase):
    r""" Calcucate min, average and maximun size of objects on the screen ."""

    def __init__(self, name: str):
        r"""
            :param name: str
                    name of component.
        """
        super().__init__(name)
        self.__sizes = dict()
        self.__checked_ids = dict()

    def do(self, data: MetaBatch) -> MetaBatch:
        r""" Counts objects. """
        for src_name in data.get_source_names():
            meta_frames = data.get_meta_frames_by_src_name(src_name)
            for meta_frame in meta_frames:
                frame = meta_frame.get_frame()
                meta_frame.set_frame(frame)
                if meta_frame.get_meta_info(MetaName.META_MASK.value) is not None:
                    self.__update(meta_frame,
                                  meta_frame.get_frame().detach().cpu().numpy().shape, src_name)
        return data

    def __update(self, meta_frame: MetaFrame, shape: Iterable[int], source: str):
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
            self.__sizes[source] = {'labels': dict(), 'size': dict()}

        meta_mask = meta_frame.get_meta_info(MetaName.META_MASK.value)
        mask = meta_mask.get_mask()
        meta_frame = self.__calculate_sizes(mask=mask, meta_frame=meta_frame)

    def __calculate_sizes(self, mask: torch.Tensor, meta_frame: MetaFrame) -> MetaFrame:
        r""" Checks whether the object crosses the line.
            :param mask: torch.Tensor
                        bounding box.
            :param shape: tuple
                        shape of frame.
        """
        frame = meta_frame.get_frame()
        shape = meta_frame.get_frame().size()
        cv_shape = (*shape[1:], shape[0])

        cv_image = mask.detach().cpu().numpy().astype(np.uint8).reshape(256, 256)
        cv_image = cv2.erode(cv_image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        cnts, _ = cv2.findContours(cv_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = list(filter(lambda cnt: cv2.contourArea(cnt) > 20, cnts))
        shape = cv_image.shape

        sizes = [cv2.contourArea(cnt) for cnt in cnts]
        try:
            min_size = round(min(sizes), 2)
            max_size = round(max(sizes), 2)
            avg_size = round(sum(sizes) / len(sizes), 2)
        except ValueError:
            min_size = 0
            max_size = 0
            avg_size = 0

        frame = frame.detach().cpu()
        frame = frame.permute(1, 2, 0).numpy()
        frame = np.ascontiguousarray(frame)
        frame = cv2.putText(frame, f"Min: {min_size}",
                            color=(255, 255, 255), fontScale=0.7, thickness=1,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            org=(50, 50))

        frame = cv2.putText(frame, f"Max: {max_size}",
                            color=(255, 255, 255), fontScale=0.7, thickness=1,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            org=(50, 70))
        frame = cv2.putText(frame, f"Avg: {avg_size}",
                            color=(255, 255, 255), fontScale=0.7, thickness=1,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            org=(50, 90))

        frame = torch.tensor(frame, device=self.get_device()).permute(2, 0, 1)

        meta_frame.set_frame(frame)
        return meta_frame

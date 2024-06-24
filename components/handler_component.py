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
            :param lines: List[int]
                    the line along which the objects will be counted. Format: [x_min, y_min, x_max, y_max]
        """
        super().__init__(name)
        self.__lines = lines
        self.__label_count = dict()
        self.__checked_ids = dict()
        self.trackers = []

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
                                  frame.permute(1, 2, 0).detach().cpu().numpy(), src_name)
                if src_name in list(self.__label_count.keys()):
                    meta_frame.add_meta('counter', self.__label_count[src_name])
        return data

    def __update(self, meta_bbox: MetaBBox, frame: np.ndarray, source: str):
        r""" Updates the current number of counted objects.
            :param meta_bbox: MetaBBox
                            metadata about the bounding boxes for the frame.
            :param frame: np.ndarray
                            frame data.
            :param source: str
                            the source from which the frame was received.
        """
        if source not in self.__checked_ids:
            self.__checked_ids[source] = set()
            self.__label_count[source] = {'labels': dict(), 'ids': dict()}

        checked_ids = list()
        bboxes = meta_bbox.get_bbox()
        label_info = meta_bbox.get_label_info()

        object_ids = label_info.get_object_ids()
        if object_ids is None or len(object_ids) == 0:
            object_ids = list(range(len(label_info.get_labels())))

        labels = label_info.get_labels()
        shape = frame.shape

        # Initialize trackers if not already done
        if not self.trackers:
            for i in range(len(bboxes)):
                tracker = cv2.TrackerKCF_create()
                x1, y1, x2, y2 = self.__bbox_denormalize_single(bboxes[i], shape)
                tracker.init(frame, (x1, y1, x2-x1, y2-y1))
                self.trackers.append((tracker, object_ids[i], labels[i]))

        new_trackers = []
        for tracker, object_id, label in self.trackers:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                center_x, center_y = x + w // 2, y + h // 2
                for line in self.__lines:
                    is_intersect = self.__check_intersect(center_x, center_y, line)
                    if is_intersect:
                        if object_id not in self.__checked_ids[source]:
                            self.__checked_ids[source].add(object_id)
                            if label not in self.__label_count[source]['labels']:
                                self.__label_count[source]['labels'][label] = 0
                            self.__label_count[source]['labels'][label] += 1
                            self.__label_count[source]['ids'][object_id] = 1
                        checked_ids.append(object_id)
                new_trackers.append((tracker, object_id, label))

        self.trackers = new_trackers

        # Update checked_ids to remove objects no longer in frame
        self.__checked_ids[source].intersection_update(checked_ids)

    def __draw_line(self, frame: torch.Tensor):
        r""" Draws a line along which objects are counted.

            :param frame: torch.Tensor
                        the frame on which the line will be drawn.
        """
        frame = frame.detach().cpu()
        frame = frame.permute(1, 2, 0).numpy()
        frame = np.ascontiguousarray(frame)

        for line in self.__lines:
            frame = cv2.line(frame, tuple(line[0]), tuple(line[1]), color=tuple(line[2]), thickness=line[3])

        return torch.tensor(frame, device=self.get_device()).permute(2, 0, 1)

    def __check_intersect(self, center_x: int, center_y: int, line) -> bool:
        r""" Checks whether the object's center crosses the line.
            :param center_x: int
                        x-coordinate of the object's center.
            :param center_y: int
                        y-coordinate of the object's center.
            :param line: tuple
                        coordinates of the line.
        """
        x1, y1 = line[0]
        x2, y2 = line[1]
        # Line equation coefficients A, B, C for the line Ax + By + C = 0
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2

        # Check the position of the center relative to the line
        position = A * center_x + B * center_y + C
        # We assume the line is horizontal or vertical
        if abs(A) > abs(B):  # Mostly vertical line
            if y1 <= center_y <= y2 or y2 <= center_y <= y1:
                return np.abs(position) < 100
        else:  # Mostly horizontal line
            if x1 <= center_x <= x2 or x2 <= center_x <= x1:
                return np.abs(position) < 100

        return False

    def __bbox_denormalize_single(self, bbox: torch.tensor, shape: Iterable[int]):
        r""" Gets coordinates for a single bounding box.
            :param bbox: torch.tensor
                        bounding box. shape: [4]
            :param shape: Iterable[int]
                        frame resolution
        """
        bbox = bbox.clone()
        bbox[0] = int(bbox[0] * shape[1])
        bbox[1] = int(bbox[1] * shape[0])
        bbox[2] = int(bbox[2] * shape[1])
        bbox[3] = int(bbox[3] * shape[0])
        return bbox.int().numpy()

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
                frame_np = frame.detach().cpu().numpy()
                frame_np = np.transpose(frame_np, (1, 2, 0))  # Преобразуем в (height, width, channels)
                for tracker, pos_before in self.__trackers:
                    success, box = tracker.update(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
                    if success:
                        centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

                meta_frame.set_frame(frame)
                if meta_frame.get_meta_info(MetaName.META_BBOX.value) is not None:
                    self.__update(meta_frame.get_meta_info(MetaName.META_BBOX.value),
                                  frame_np,
                                  self.base_speed)
        return data

    def __update(self, meta_bbox: MetaBBox, frame, base_speed):
        r""" Updates the current number of counted objects.
            :param meta_bbox: MetaBBox
                            metadata about the bounding boxes for the frame.
            :param frame: ndarray
                            frame data in (height, width, channels).
            :param base_speed: float
                            base speed for calculations.
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
                                    (box[1] + box[3]) / 2 * frame.shape[0]]
                    track_pos_before = self.__trackers[n][1]
                    success, track_box = self.__trackers[n][0].update(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if success:
                        track_pos = ((track_box[0] + track_box[2]) / 2, (track_box[1] + track_box[3]) / 2)
                        speed = ((track_pos_before[0] - track_pos[0]) ** 2 + (track_pos_before[1] - track_pos[1]) ** 2) ** 0.5
                        if base_speed is not None:
                            speed = base_speed + speed / 50_000 + math.sin(speed) * 10
                        new_labels[n] = labels[n] + f'-{int(speed)}kph'
                        self.__trackers[n][1] = track_pos
                except Exception as e:
                    print(e)
        confs = label_info.get_confidence()
        meta_bbox.set_label_info(MetaLabel(new_labels, confs))
        self.__trackers = []
        for i, conf in zip(bboxes, confs):
            tracker = cv2.TrackerKCF_create()
            x1 = int(i[0] * frame.shape[1])
            y1 = int(i[1] * frame.shape[0])
            x2 = int(i[2] * frame.shape[1])
            y2 = int(i[3] * frame.shape[0])

            # Корректировка координат, чтобы они находились в пределах изображения
            x1 = max(0, min(x1, frame.shape[1] - 1))
            y1 = max(0, min(y1, frame.shape[0] - 1))
            x2 = max(0, min(x2, frame.shape[1] - 1))
            y2 = max(0, min(y2, frame.shape[0] - 1))

            # Дополнительная проверка и корректировка, чтобы высота и ширина были корректны
            if x2 <= x1:
                x2 = x1 + 1
            if y2 <= y1:
                y2 = y1 + 1

            # Проверка, что ширина и высота не равны нулю
            if x2 - x1 > 0 and y2 - y1 > 0:
                rect = (x1, y1, x2 - x1, y2 - y1)
                # Отладочная информация
                print(f"Initializing tracker with rect: {rect}, frame shape: {frame.shape}")
                tracker.init(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), rect)
                self.__trackers.append([tracker, (x1, y1)])
            else:
                print(f"Invalid bounding box dimensions after correction: {(x1, y1, x2 - x1, y2 - y1)}")


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

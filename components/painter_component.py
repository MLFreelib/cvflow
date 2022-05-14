import os
import random
from typing import Union, List

import cv2
import numpy as np
import torch
import torchvision.transforms
from torchvision.transforms import Resize
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks, make_grid

from Meta import MetaFrame, MetaBatch
from components.component_base import ComponentBase


def _generate_color() -> tuple: return tuple([random.randint(0, 255) for _ in range(3)])


class Painter(ComponentBase):
    r""" Component of basic painter """
    pass


class Tiler(Painter):
    r""" Component which combines frames from different sources into one frame in the form of a grid.
        :param name: str
                    name of component
        :param tiler_size: tuple
                    number of rows and columns. Example: (3, 2) for 6 frames. If there are not enough frames,
                    then the remaining space is filled black.
    """

    def __init__(self, name: str, tiler_size: tuple):
        super().__init__(name)
        self.__tile_size = (360, 640)
        self.__tiler_size = tiler_size

    def do(self, data: Union[MetaBatch, MetaFrame]) -> Union[MetaBatch, MetaFrame]:
        r""" Combines frames from different sources into one frame in the form of a grid. """
        frames = data.get_meta_frames_all()

        tiles = list()
        sources = data.get_source_names()

        for source in sources:
            meta_frames = frames[source]
            for i in range(len(meta_frames)):
                frame = meta_frames[i].get_frame()
                frame = Resize(self.__tile_size).forward(frame)
                frame = torch.unsqueeze(frame.cpu(), 0)
                if (i + 1) > len(tiles):
                    tiles.append(frame)
                else:
                    tiles[i] = torch.cat((tiles[i], frame), dim=0)

        for i in range(len(tiles)):
            tile = make_grid(tiles[i], nrow=self.__tiler_size[0], padding=0)
            frame = MetaFrame('tiler', tile)
            data.add_meta_frame(frame)
        return data

    def set_size(self, size: tuple):
        r""" Resolution of the output frame.
            :param size: tuple
                    resolution. Example: (1280, 1920)
        """
        if len(size) != 2:
            raise ValueError(f'Expected length of size 2, actual {len(size)}')
        self.__tile_size = (size[0] // self.__tiler_size[0], size[1] // self.__tiler_size[1])


class BBoxPainter(Painter):
    r""" A component for drawing bounding boxes on frames.
        :param name: str
                    name of component
        :param font_path: str
                    path to font
        :param font_size: int
                    font size
        :param font_width: int
                    font width
    """

    def __init__(self, name: str, font_path: str, font_size: int = 20, font_width: int = 3):
        super().__init__(name)
        self.__font_path = font_path
        self.__font_size = font_size
        self.__font_width = font_width
        self.__colors = dict()
        self.__resolution = None

    def set_font_size(self, font_size: int):
        if isinstance(font_size, int):
            self.__font_size = font_size

    def set_font_width(self, font_width: int):
        if isinstance(font_width, int):
            self.__font_width = font_width

    def do(self, data: Union[MetaBatch, MetaFrame]) -> Union[MetaBatch, MetaFrame]:
        r""" Draws bounding boxes with labels on frames. """

        for source in data.get_source_names():
            for meta_frame in data.get_meta_frames_by_src_name(source):
                shape = meta_frame.get_frame().shape
                if meta_frame.get_bbox_info() is not None:
                    meta_bbox = meta_frame.get_bbox_info()
                    bbox = meta_bbox.get_bbox()
                    self.__bbox_denormalize(bbox, shape)
                    meta_labels = meta_bbox.get_label_info()
                    ids = meta_labels.get_object_ids()
                    labels = meta_labels.get_labels()
                    if len(ids) == 0:
                        meta_objects_info = zip(labels, meta_labels.get_confidence())
                        full_labels = [f'{label} {round(conf * 100)}%' for label, conf in meta_objects_info]
                    else:
                        meta_objects_info = zip(labels, meta_labels.get_confidence(), ids)
                        full_labels = [f'{obj_id} {label} {round(conf * 100)}%' for label, conf, obj_id in
                                       meta_objects_info]
                    frame = meta_frame.get_frame().cpu()
                    if self.__resolution is None:
                        self.__resolution = frame.shape[-2:]

                    frame = torchvision.transforms.Resize(self.__resolution)(frame)
                    bboxes_frame = draw_bounding_boxes(frame,
                                                       boxes=bbox,
                                                       width=self.__font_width,
                                                       labels=full_labels,
                                                       font_size=self.__font_size,
                                                       font=self.__font_path,
                                                       colors=self.__get_colors(labels))

                    counter = meta_frame.get_meta_info('counter')
                    if counter is not None:
                        bboxes_frame = self.___put_count(bboxes_frame, meta_frame.get_meta_info('counter'))
                    meta_frame.set_frame(bboxes_frame)
        return data

    def ___put_count(self, frame: torch.Tensor, counts: dict):
        if len(list(counts['labels'].keys())) == 0:
            return frame
        frame = frame.detach().cpu()
        frame = frame.permute((1, 2, 0)).numpy()
        frame = np.ascontiguousarray(frame)
        for i in range(len(counts['labels'].keys())):
            label = list(counts["labels"].keys())[i]
            frame = cv2.putText(frame,
                                text=f'Count of {label}: {str(counts["labels"][label])}',
                                org=(50, (1 + i) * 30), fontFace=0,
                                color=(255, 0, 0), thickness=2, lineType=16,
                                fontScale=0.5)
        return torch.tensor(frame, device=self.get_device()).permute((2, 0, 1))

    def start(self):
        r""" Checks types and sets default values. """

        if self.__font_path is None:
            raise FileNotFoundError(f'Font is a required parameter')

        if not os.path.exists(self.__font_path):
            raise FileNotFoundError(f'Font {self.__font_path} not found')

        _, file_extension = os.path.splitext(self.__font_path)
        if file_extension != '.ttf':
            raise ValueError(f'Expected .ttf, actual {file_extension}')

        if self.__font_width <= 0:
            self.__font_width = 5

        if self.__font_size <= 0:
            self.__font_size = 20

    def __bbox_denormalize(self, bboxes: torch.tensor, shape: torch.tensor):
        r""" Gets coordinates for bounding boxes.
            :param bboxes: torch.tensor
                        bounding boxes. shape: [N, 4]
            :param shape: torch.tensor
                        frame resolution
        """
        bboxes[:, (0, 2)] = bboxes[:, (0, 2)].mul(shape[2])
        bboxes[:, (1, 3)] = bboxes[:, (1, 3)].mul(shape[1])

    def __get_colors(self, labels: List) -> List:
        colors = list()
        for label_name in labels:
            if label_name not in list(self.__colors.keys()):
                self.__colors[label_name] = _generate_color()
            colors.append(self.__colors[label_name])
        return colors


class LabelPainter(Painter):
    r""" Writes a label to an image.
        :param name: str
                   name of component
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.__font_face = 0
        self.__org = (30, 30)
        self.__colors = dict()
        self.__thickness = 2
        self.__lineType = 16
        self.__font_scale = 1
        self.__resolution = None

    def set_font_face(self, font_face: int):
        if isinstance(font_face, int):
            self.__font_face = font_face

    def set_org(self, org: tuple):
        if isinstance(org, tuple):
            if len(org) == 2:
                self.__org = org

    def set_colors(self, colors: dict):
        if isinstance(colors, dict):
            self.__colors = colors

    def set_thickness(self, thickness: int):
        if isinstance(thickness, int):
            self.__thickness = thickness

    def set_lineType(self, lineType: int):
        if isinstance(lineType, int):
            self.__lineType = lineType

    def set_font_scale(self, font_scale: int):
        if isinstance(font_scale, int):
            self.__font_scale = font_scale

    def do(self, data: Union[MetaBatch, MetaFrame]) -> Union[MetaBatch, MetaFrame]:
        r""" Writes labels on the frames. """
        for source in data.get_source_names():
            for meta_frame in data.get_meta_frames_by_src_name(source):
                label_info = meta_frame.get_labels_info()
                if meta_frame.get_labels_info() is not None:
                    labels = label_info.get_labels()
                    label_confidence = label_info.get_confidence()
                    label_id = torch.max(label_confidence, dim=1)[1]
                    label_id = label_id.detach().cpu().numpy()[0]
                    label_name = labels[label_id]

                    frame = meta_frame.get_frame()
                    frame = frame.detach().cpu()
                    frame = frame.permute((1, 2, 0)).numpy()
                    if self.__resolution is None:
                        self.__resolution = frame.shape[1], frame.shape[0]
                    frame = cv2.resize(frame, self.__resolution)
                    frame = np.ascontiguousarray(frame)
                    frame = cv2.putText(frame,
                                        text=f'{label_name}',
                                        org=self.__org, fontFace=self.__font_face,
                                        color=self.__get_label_color(label_name), thickness=2, lineType=self.__lineType,
                                        fontScale=self.__font_scale)
                    frame = torch.tensor(frame, device=self.get_device())
                    frame = frame.permute((2, 0, 1))
                    meta_frame.set_frame(frame)

        return data

        return frame

    def __get_label_color(self, label_name: str):
        if label_name not in self.__colors.keys():
            self.__colors[label_name] = _generate_color()
        return self.__colors[label_name]


class MaskPainter(Painter):
    r"""A component for drawing masks on frames.
        :param name: str
                   name of component
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.__colors = dict()
        self.__alpha = 0.8

    def set_alpha(self, alpha: float):
        if isinstance(alpha, float):
            if 0 <= alpha <= 1:
                self.__alpha = alpha

    def do(self, data: MetaBatch) -> MetaBatch:
        r""" Draws masks on frames. """
        for source in data.get_source_names():
            for meta_frame in data.get_meta_frames_by_src_name(source):
                meta_mask = meta_frame.get_mask_info()
                masks = meta_mask.get_mask()
                frame = meta_frame.get_frame()
                colors = self.__get_colors(meta_mask.get_label_info().get_labels())
                for mask in masks:
                    resized_mask = torchvision.transforms.Resize((frame.shape[-2:]))(mask)
                    frame = draw_segmentation_masks(frame.detach().cpu(), resized_mask.detach().cpu(),
                                                    alpha=self.__alpha,
                                                    colors=colors)
                meta_frame.set_frame(frame)
        return data

    def __get_colors(self, labels: List):
        colors = list()
        for label_name in labels:
            if label_name not in self.__colors.keys():
                self.__colors[label_name] = _generate_color()
            colors.append(self.__colors[label_name])
        return colors

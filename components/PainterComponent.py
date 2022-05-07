import os
from typing import Union

import cv2
import torch
from torchvision.transforms import Resize
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks, make_grid

from Meta import MetaFrame, MetaBatch
from components.ComponentBase import ComponentBase


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
        self.__dividers = None

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
            tile = make_grid(tiles[i], nrow=2)
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

    def do(self, data: Union[MetaBatch, MetaFrame]) -> Union[MetaBatch, MetaFrame]:
        r""" Draws bounding boxes with labels on frames. """

        for source in data.get_source_names():
            for frame in data.get_meta_frames_by_src_name(source):
                shape = frame.get_frame().shape
                if frame.get_bbox_info() is not None:
                    meta_bbox = frame.get_bbox_info()
                    bbox = meta_bbox.get_bbox()
                    self.__bbox_denormalize(bbox, shape)
                    meta_labels = meta_bbox.get_label_info()
                    ids = meta_labels.get_object_ids()
                    labels = meta_labels.get_labels()
                    if len(ids) == 0:
                        meta_objects_info = zip(labels, meta_labels.get_confidence())
                        labels = [f'{label} {round(conf * 100)}%' for label, conf in meta_objects_info]
                    else:
                        meta_objects_info = zip(labels, meta_labels.get_confidence(), ids)
                        labels = [f'{obj_id} {label} {round(conf * 100)}%' for label, conf, obj_id in meta_objects_info]

                    bboxes_frame = draw_bounding_boxes(frame.get_frame().cpu(),
                                                       boxes=bbox,
                                                       width=self.__font_width,
                                                       labels=labels,
                                                       font_size=self.__font_size,
                                                       font=self.__font_path)
                    frame.set_frame(bboxes_frame)
        return data

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


class LabelPainter(Painter):
    r""" Writes a label to an image.

        :param name: str
                   name of component
    """
    def __init__(self, name: str):
        super().__init__(name)

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
                    frame = cv2.putText(frame,
                                        text=f'{label_name}',
                                        org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        color=(128, 128, 64), thickness=2, lineType=cv2.LINE_AA, fontScale=1)

                    frame = torch.tensor(frame, device=self.get_device())
                    frame = frame.permute((2, 0, 1))
                    meta_frame.set_frame(frame)

        return data


class MaskPainter(Painter):
    r"""A component for drawing masks on frames.

        :param name: str
                   name of component
    """
    def __init__(self, name: str):
        super().__init__(name)

    def do(self, data: MetaBatch) -> MetaBatch:
        r""" Draws masks on frames. """
        for source in data.get_source_names():
            for meta_frame in data.get_meta_frames_by_src_name(source):
                meta_mask = meta_frame.get_mask_info()
                masks = meta_mask.get_mask()
                frame = meta_frame.get_frame()
                for mask in masks:
                    frame = draw_segmentation_masks(frame.detach().cpu(), mask.detach().cpu())
                meta_frame.set_frame(frame)
        return data



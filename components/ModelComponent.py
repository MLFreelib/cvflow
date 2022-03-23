import time
from typing import Union, List

import numpy as np
import torch

from Meta import MetaBatch, MetaFrame, MetaLabel, MetaBBox
from components.ComponentBase import ComponentBase
from exceptions import MethodNotOverriddenException


class ModelBase(ComponentBase):

    def __init__(self, name: str, model: torch.nn.Module):
        super().__init__(name)
        self.__transforms = list()
        self._connected_sources = list()
        self._frame_resolution: tuple = (480, 640)
        self._inference: torch.nn.Module = model
        self._inference.eval()
        self._confidence = 0.8

    def set_confidence(self, conf: float):
        self._confidence = conf

    def set_resolution(self, frame_resolution):
        self._frame_resolution = frame_resolution

    def do(self, data: Union[MetaBatch, MetaFrame]):
        raise MethodNotOverriddenException('read in the ReaderBase')

    def start(self):
        self._inference.to(device=self.get_device())

    def set_transforms(self, tensor_transforms: list):
        self.__transforms = tensor_transforms

    def add_source(self, name: str):
        self._connected_sources.append(name)

    def _get_transforms(self):
        return self.__transforms

    def _transform(self, data: torch.Tensor):
        for t_transform in self._get_transforms():
            data = t_transform.forward(data)
        return data


class ModelClassifier(ModelBase):
    def __init__(self, name: str, label_names: Union[list, tuple], model: torch.nn.Module):
        super().__init__(name, model)
        self.__label_names = label_names

    def do(self, data: Union[MetaBatch, MetaFrame]):
        for src_name in self._connected_sources:
            needed_data = data.get_frames_by_src_name(src_name)
            if needed_data is None:
                continue

            needed_data = needed_data.clone().to(dtype=torch.float, device=self.get_device())
            self._transform(needed_data)

            with torch.no_grad():
                probabilities = self._inference(needed_data)
                probabilities = torch.nn.functional.softmax(probabilities, dim=1)[0] * 100

            label = self.__label_names[probabilities.cpu().detach().numpy().argmax()]
        return data

    def __set_labels(self, data: list, labels: torch.tensor, confidence: float):
        for meta_frame in data:
            label_id = labels.cpu().detach().numpy().argmax()
            meta_label = MetaLabel(label=self.__label_names[label_id], confidence=confidence)
            meta_label.set_object_id(label_id)
            meta_frame.add_label_info(meta_label)


class ModelDetection(ModelBase):
    def __init__(self, name: str, model: torch.nn.Module):
        super().__init__(name, model)
        self.__label_names = None

    def set_labels(self, labels: List[str]):
        self.__label_names = labels

    #
    # def do(self, data: Union[MetaBatch, MetaFrame]):
    #     t1 = time.time()
    #     for src_name in self._connected_sources:
    #         needed_data = data.get_frames_by_src_name(src_name)
    #         if needed_data is None:
    #             continue
    #
    #         cloned_data = needed_data.clone().to(dtype=torch.float, device=self.get_device())
    #         self._transform(cloned_data)
    #
    #         cloned_data = cloned_data.div(255)
    #
    #         with torch.no_grad():
    #             preds = self._inference(cloned_data)
    #
    #         self.__to_meta(data=data, preds=preds, shape=cloned_data.shape, src_name=src_name)
    #
    #     print(time.time() - t1)
    #     return data
    #

    def do(self, data: Union[MetaBatch, MetaFrame]):
        src_data = list()
        for src_name in self._connected_sources:
            needed_data = data.get_frames_by_src_name(src_name)
            if needed_data is None:
                continue

            cloned_data = needed_data.clone().to(dtype=torch.float, device=self.get_device())
            cloned_data = self._transform(cloned_data)

            cloned_data = cloned_data.div(255)

            src_data.append(cloned_data)

        batch, src_size = self.to_tensor(src_data)

        with torch.no_grad():
            preds = self._inference(batch)

        i_point = 0
        for i_src_name in range(len(self._connected_sources)):
            src_name = self._connected_sources[i_src_name]
            shape = src_data[i_src_name].shape
            self.__to_meta(data=data, preds=preds[i_point: i_point + src_size[i_src_name]], shape=shape, src_name=src_name)
            i_point += src_size[i_src_name]

        return data

    def to_tensor(self, src_data):
        batch = torch.cat(src_data, dim=0)
        size_frames = list()
        for frames in src_data:
            size_frames.append(frames.shape[0])
        return batch, size_frames

    def __to_meta(self, data: MetaBatch, preds: torch.Tensor, shape: torch.Tensor, src_name: str):
        for i in range(len(preds)):
            boxes = preds[i]['boxes'].cpu()
            labels = preds[i]['labels'].cpu().detach().numpy()
            conf = preds[i]['scores'].cpu().detach().numpy()
            true_conf = conf > self._confidence
            if np.any(true_conf):
                conf = conf[true_conf]
                boxes = boxes[true_conf]
                label_names = [self.__label_names[ind] for ind in labels[true_conf]]

                self.__bbox_normalize(boxes, shape)
                meta_frame = data.get_meta_frames_by_src_name(src_name)[i]
                meta_label = MetaLabel(labels=label_names, confidence=conf)
                meta_label.get_confidence()
                meta_frame.set_label_info(meta_label)
                meta_frame.set_bbox_info(MetaBBox(boxes, meta_label))

    def __bbox_normalize(self, bboxes: torch.tensor, shape: torch.tensor):
        # [N, xmin, ymin, xmax, ymax] | [N, C, H, W]
        bboxes[:, (0, 2)] = bboxes[:, (0, 2)].div(shape[3])
        bboxes[:, (1, 3)] = bboxes[:, (1, 3)].div(shape[2])

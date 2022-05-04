from typing import Union, List

import numpy as np
import torch

from Meta import MetaBatch, MetaFrame, MetaLabel, MetaBBox, MetaMask
from components.ComponentBase import ComponentBase
from exceptions import MethodNotOverriddenException


def _to_tensor(src_data):
    batch = torch.cat(src_data, dim=0)
    size_frames = list()
    for frames in src_data:
        size_frames.append(frames.shape[0])
    return batch, size_frames


def _to_model(connected_sources: List[str], data: MetaBatch, device: str, transform):
    src_data = list()
    for src_name in connected_sources:
        needed_data = data.get_frames_by_src_name(src_name)
        if needed_data is None:
            continue

        cloned_data = needed_data.clone().to(dtype=torch.float, device=device)
        cloned_data = transform(cloned_data)

        cloned_data = cloned_data.div(255)

        src_data.append(cloned_data)
    return src_data


class ModelBase(ComponentBase):

    def __init__(self, name: str, model: torch.nn.Module):
        super().__init__(name)
        self.__transforms = list()
        self._connected_sources = list()
        self._frame_resolution: tuple = (480, 640)
        self._inference: torch.nn.Module = model
        self._inference.eval()
        self._confidence = 0.8
        self._label_names = None

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

    def set_labels(self, labels: List[str]):
        self._label_names = labels

    def _get_transforms(self):
        return self.__transforms

    def _transform(self, data: torch.Tensor):
        if self._get_transforms() is not None:
            for t_transform in self._get_transforms():
                data = t_transform.forward(data)
        return data


class ModelDetection(ModelBase):
    def __init__(self, name: str, model: torch.nn.Module):
        super().__init__(name, model)

    def do(self, data: Union[MetaBatch, MetaFrame]):
        src_data = _to_model(connected_sources=self._connected_sources,
                             data=data,
                             device=self.get_device(),
                             transform=self._transform)

        batch, src_size = _to_tensor(src_data)

        with torch.no_grad():
            preds = self._inference(batch)

        i_point = 0
        for i_src_name in range(len(self._connected_sources)):
            src_name = self._connected_sources[i_src_name]
            shape = src_data[i_src_name].shape
            self.__to_meta(data=data, preds=preds[i_point: i_point + src_size[i_src_name]], shape=shape,
                           src_name=src_name)
            i_point += src_size[i_src_name]

        return data

    def __to_meta(self, data: MetaBatch, preds: torch.Tensor, shape: torch.Tensor, src_name: str):
        for i in range(len(preds)):
            boxes = preds[i]['boxes'].cpu()
            labels = preds[i]['labels'].cpu().detach().numpy()
            conf = preds[i]['scores'].cpu().detach().numpy()
            true_conf = conf > self._confidence
            if np.any(true_conf):
                conf = conf[true_conf]
                boxes = boxes[true_conf]
                label_names = [self._label_names[ind] for ind in labels[true_conf]]

                self.__bbox_normalize(boxes, shape)
                meta_frame = data.get_meta_frames_by_src_name(src_name)[i]
                meta_label = MetaLabel(labels=label_names, confidence=conf)

                meta_frame.set_bbox_info(MetaBBox(boxes, meta_label))

    def __bbox_normalize(self, bboxes: torch.tensor, shape: torch.tensor):
        # [N, xmin, ymin, xmax, ymax] | [N, C, H, W]
        bboxes[:, (0, 2)] = bboxes[:, (0, 2)].div(shape[3])
        bboxes[:, (1, 3)] = bboxes[:, (1, 3)].div(shape[2])


class ModelClassification(ModelBase):
    def __init__(self, name: str, model: torch.nn.Module):
        super().__init__(name, model)

    def do(self, data: Union[MetaBatch, MetaFrame]) -> MetaBatch:
        src_data = _to_model(connected_sources=self._connected_sources,
                             data=data,
                             device=self.get_device(),
                             transform=self._transform)

        batch, src_size = _to_tensor(src_data)

        with torch.no_grad():
            probabilities = self._inference(batch)

        prob_i = 0
        for src_name in self._connected_sources:
            for src_frames_size in src_size:
                for i in range(src_frames_size):
                    meta_frame = data.get_meta_frames_by_src_name(src_name)[i]
                    probability = probabilities[prob_i]
                    probability = probability[None, :]
                    meta_label = MetaLabel(labels=self._label_names, confidence=probability)
                    meta_frame.set_label_info(meta_label)
                    prob_i += 1
        return data


class ModelSegmentation(ModelBase):

    def __init__(self, name: str, model: torch.nn.Module):
        super().__init__(name, model)

    def do(self, data: MetaBatch) -> MetaBatch:
        src_data = _to_model(connected_sources=self._connected_sources,
                             data=data,
                             device=self.get_device(),
                             transform=self._transform)

        batch, src_size = _to_tensor(src_data)

        with torch.no_grad():
            output = self._inference(batch)['out']

        normalized_masks = torch.nn.functional.softmax(output, dim=1)

        prob_i = 0
        for src_name in self._connected_sources:
            for src_frames_size in src_size:
                for i in range(src_frames_size):
                    meta_frame = data.get_meta_frames_by_src_name(src_name)[i]
                    normalized_mask = normalized_masks[prob_i]
                    mask = torch.zeros(normalized_mask.shape, dtype=torch.bool, device=self.get_device())
                    mask[normalized_mask > self._confidence] = True
                    mask = mask[None, :]
                    meta_mask = MetaMask(mask, MetaLabel(self._label_names, normalized_mask))
                    meta_frame.set_mask_info(meta_mask)
                    prob_i += 1
        return data


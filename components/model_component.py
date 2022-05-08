from typing import List

import numpy as np
import torch

from Meta import MetaBatch, MetaLabel, MetaBBox, MetaMask
from components.component_base import ComponentBase


def _to_tensor(src_data: List[torch.tensor]):
    r""" Combines frames from different sources into one batch.

        :param src_data: List[torch.tensor] - list of tensors with frames. Tensor shape: [N, C, W, H], where N - batch
         size, C - channels, W - width, H - height
        :return: batch: Tensor, batch of frames
                 size_frames: number of frames for each source.

                Example:
                    size_frames is [3, 2, 3] and batch size is 8 means that the first 3 frames in the batch
                    belong to the first source, the next 2 frames belong to the second source, and the last 3 belong
                    to the last source.
    """

    batch = torch.cat(src_data, dim=0)
    size_frames = list()
    for frames in src_data:
        size_frames.append(frames.shape[0])
    return batch, size_frames


def _to_model(connected_sources: List[str], data: MetaBatch, device: str, transform) -> List[torch.tensor]:
    r""" Returns a list of transformed frames from the MetaBatch.
        :param connected_sources:
        :param data: MetaData
        :param device: str - cuda or cpu
        :param transform: list of transformations from torch.transform
        :return: List[torch.tensor]
    """
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
    r""" Component of basic model. This class is necessary for implementing models using inheritance.

        :param name: str
                name of component.
        :param model: torch.nn.Module
    """

    def __init__(self, name: str, model: torch.nn.Module):
        super().__init__(name)
        self.__transforms = list()
        self._connected_sources = list()
        self._inference: torch.nn.Module = model
        self._inference.eval()
        self._confidence = 0.8
        self.__label_names = None

    def set_confidence(self, conf: float):
        """ Setting the confidence threshold.
            :param conf: float [0-1]
        """
        if conf > 1:
            self._confidence = 1
        elif conf < 0:
            self._confidence = 0

    def start(self):
        r""" Specifies the device on which the model will be executed. """
        self._inference.to(device=self.get_device())

    def set_transforms(self, tensor_transforms: list):
        r""" Method of setting transformations for frames that are passed to the model.
            :param tensor_transforms: list of transformations from torchvision.transforms
        """
        self.__transforms = tensor_transforms

    def add_source(self, name: str):
        r""" Names of input sources from which data will be processed by the component.
            :param name: str
                    name of source
        """
        self._connected_sources.append(name)

    def set_labels(self, labels: List[str]):
        r""" Sets labels for model
            :param labels: List[str]
                    list of labels
        """
        self.__label_names = labels

    def get_labels(self):
        r""" Returns the label names. """
        return self.__label_names

    def _get_transforms(self):
        r""" Returns a list of transformations. """
        return self.__transforms

    def _transform(self, data: torch.tensor):
        r""" Transforms the data.
            :param data: torch.tensor
        """
        if self._get_transforms() is not None:
            for t_transform in self._get_transforms():
                data = t_transform.forward(data)
        return data


class ModelDetection(ModelBase):
    r""" Component for detection models.
        The model must have a forward method that returns a dictionary with the keys:
            - boxes - [N, 4]
            - labels - [N]
            - scores - [N]

        Examples:
            Example for the forward method of the model:

                def forward(self, x):
                    ...
                    return {boxes:...,
                            labels: ...,
                            scores: ...}
            Example of data:
                - boxes: tensor([[12, 15, 53, 74],
                          [101, 56, 156, 89],
                          41, 32, 112, 96]]), where shape [3, 4]
                - labels: tensor([0, 3, 3])
                - scores: tensor([0.832, 0.12, 0.675])

    """

    def __init__(self, name: str, model: torch.nn.Module):
        super().__init__(name, model)

    def do(self, data: MetaBatch) -> MetaBatch:
        r""" Transmits data to the detection model. And adds the predicted bounding boxes with labels to the MetaFrame
            in the MetaBatch. Bounding boxes in BBoxMeta and labels in MetaLabel, which are contained in BBoxMeta.
        """
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
            shape = src_data[i_src_name].shape[-2:]
            self.__to_meta(data=data, preds=preds[i_point: i_point + src_size[i_src_name]], shape=shape,
                           src_name=src_name)
            i_point += src_size[i_src_name]

        return data

    def __to_meta(self, data: MetaBatch, preds: list, shape: torch.Tensor, src_name: str):
        r""" Adds bounding boxes to MetaBatch.
            :param data: MetaBatch
            :param preds: A list of floating point values.
            :param shape: torch.tensor - image resolution.
            :param src_name: str - source name
        """
        for i in range(len(preds)):
            boxes = preds[i]['boxes'].cpu()
            labels = preds[i]['labels'].cpu().detach().numpy()
            conf = preds[i]['scores'].cpu().detach().numpy()
            true_conf = conf > self._confidence
            if np.any(true_conf):
                conf = conf[true_conf]
                boxes = boxes[true_conf]
                label_names = [self.get_labels()[ind] for ind in labels[true_conf]]

                self.__bbox_normalize(boxes, shape)
                meta_frame = data.get_meta_frames_by_src_name(src_name)[i]
                meta_label = MetaLabel(labels=label_names, confidence=conf)

                meta_frame.set_bbox_info(MetaBBox(boxes, meta_label))

    def __bbox_normalize(self, bboxes: torch.tensor, shape: torch.tensor):
        r""" Normalization of bounding box values in the range from 0 to 1.
            :param bboxes: torch.tensor
            :param shape: torch.tensor - image resolution.
            :return:
        """
        bboxes[:, (0, 2)] = bboxes[:, (0, 2)].div(shape[1])
        bboxes[:, (1, 3)] = bboxes[:, (1, 3)].div(shape[0])
        return bboxes


class ModelClassification(ModelBase):
    r""" Component for classification models

        :param name: str
                    name of component

        :param model: torch.nn.Module
                    classification model, which returns vector of shape [N, K], where N - batch size, K - number of labels
                    and values in the range from 0 to 1.
    """

    def __init__(self, name: str, model: torch.nn.Module):
        super().__init__(name, model)

    def do(self, data: MetaBatch) -> MetaBatch:
        r""" Transmits data to the classification model. And adds the predicted labels to the MetaFrame
                   in the MetaBatch. Labels in MetaLabel, which are contained in MetaFrame.
               """
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
                    meta_label = MetaLabel(labels=self.get_labels(), confidence=probability)
                    meta_frame.set_label_info(meta_label)
                    prob_i += 1
        return data


class ModelSegmentation(ModelBase):
    r""" Component for segmentation models

        :param name: str
                name of component

        :param model: torch.nn.Module
                    segmentation model, which returns dictionary with key "out" which contains tensor of shape
                    [N, K, H, W], where N - batch size, K - number of labels, H - mask height, W - mask width and
                    values in the range from 0 to 1.
    """

    def __init__(self, name: str, model: torch.nn.Module):
        super().__init__(name, model)

    def do(self, data: MetaBatch) -> MetaBatch:
        r""" Transmits data to the segmentation model. And adds the predicted masks with labels to the MetaFrame
            in the MetaBatch. Masks in MetaMask and labels in MetaLabel, which are contained in MetaMask.
        """
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
                    meta_mask = MetaMask(mask, MetaLabel(self.get_labels(), normalized_mask))
                    meta_frame.set_mask_info(meta_mask)
                    prob_i += 1
        return data

import gc
from typing import List, Tuple, Union, Any, Dict

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer

from Meta import MetaBatch, MetaLabel, MetaBBox, MetaMask, MetaDepth, MetaName
from components.component_base import ComponentBase
from models.blocks import OutputFormat
from models.losses_structures.loss_base import LossBase


class ModelBase(ComponentBase):
    r""" Component of basic model. This class is necessary for implementing models using inheritance.

        :param name: str
                name of component.
        :param model: torch.nn.Module
    """

    def __init__(self, name: str, model: torch.nn.Module):
        super().__init__(name)
        self.__transforms = list()
        self._inference: torch.nn.Module = model
        self._confidence = 0.8
        self.__label_names = None
        self.is_train = False

    def set_confidence(self, conf: float):
        """ Setting the confidence threshold.
            :param conf: float [0-1]
        """
        if conf > 1:
            self._confidence = 1
        elif conf < 0:
            self._confidence = 0
        else:
            self._confidence = conf

    def do(self, data: MetaBatch) -> MetaBatch:
        batch, src_size = self._to_model_format(connected_sources=self._source_names,
                                                data=data,
                                                device=self.get_device(),
                                                transform=self._transform)
        if self.is_train:
            predictions = self._to_train(batch, data)
            for src_name in data.get_source_names():
                for meta_frame in data.get_meta_frames_by_src_name(src_name):
                    meta_frame.add_meta('losses', predictions)
        else:
            predictions = self._to_inference(batch)
            self._add_to_meta_all(data, batch, predictions, src_size)
        return data

    def start(self):
        r""" Specifies the device on which the model will be executed. """
        self._inference.to(device=torch.device(self.get_device()))
        # self._inference.train() if self.is_train else self._inference.eval()

    def add_training_params(self, optimizer: Optimizer, loss_func: LossBase):
        self.optimizer = optimizer
        self.loss_func = loss_func

    def stop(self):
        del self._inference
        gc.collect()
        torch.cuda.empty_cache()

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
        self._source_names.append(name)

    def set_labels(self, labels: List[str]):
        r""" Sets labels for model
            :param labels: List[str]
                    list of labels
        """
        self.__label_names = labels

    def get_labels(self):
        r""" Returns the label names. """
        return self.__label_names

    def _to_model_format(self, connected_sources: List[str], data: MetaBatch, device: str, transform, *args,
                         **kwargs) -> Tuple[Tensor, List[Any]]:
        r""" Returns a list of transformed frames from the MetaBatch.
            :param connected_sources:
            :param data: MetaData
            :param device: str - cuda or cpu
            :param transform: list of transformations from torch.transform
            :return: Tuple[Tensor, List[Any]]
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
        return self._to_tensor(src_data)

    def _to_tensor(self, src_data: List[torch.tensor]):
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

    def _to_inference(self, batch: torch.Tensor, *args, **kwargs) -> Dict:
        with torch.no_grad():
            t = batch[0].numpy().reshape(640, 640, 3)
            return self._inference(t)


    def _to_train(self, batch: torch.Tensor, true_values: MetaBatch):
        self.optimizer.zero_grad()
        out = self._inference(batch)
        losses, loss = self.loss_func(out, true_values)
        loss.backward()
        return losses

    def _add_to_meta_all(self, meta_batch: MetaBatch, src_data: List, predictions, src_size, *args, **kwargs):
        pass

    def _add_to_meta(self, data: MetaBatch, predictions: list, shape: torch.Tensor, src_name: list, *args, **kwargs):
        pass


    def _get_transforms(self):
        r""" Returns a list of transformations. """
        return self.__transforms

    def _transform(self, data: torch.tensor):
        r""" Transforms the data.
            :param data: torch.tensor
        """
        if self._get_transforms() is not None:
            for t_transform in self._get_transforms():
                data = t_transform.forward(data, )
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

    def _add_to_meta_all(self, meta_batch: MetaBatch, src_data: List, predictions, src_size, *args, **kwargs):
        i_point = 0
        for i_src_name in range(len(self._source_names)):
            src_name = self._source_names[i_src_name]
            shape = src_data[i_src_name].shape[-2:]
            self._add_to_meta(data=meta_batch, preds=predictions[i_point: i_point + src_size[i_src_name]], shape=shape,
                              src_name=src_name)
            i_point += src_size[i_src_name]

    def _add_to_meta(self, data: MetaBatch, preds: list, shape: torch.Tensor, src_name: str, **kwargs):
        r""" Adds bounding boxes to MetaBatch.
            :param data: MetaBatch
            :param preds: A list of floating point values.
            :param shape: torch.tensor - image resolution.
            :param src_name: str - source name
        """
        for i in range(len(preds)):
            # print('i', i, preds[0].boxes)
            boxes = preds[i]['boxes'].cpu()
            labels = preds[i]['labels'].cpu().detach().numpy()
            conf = preds[i]['scores'].cpu().detach().numpy()
            true_conf = conf > self._confidence
            if np.any(true_conf):
                conf = conf[true_conf]
                boxes = boxes[true_conf]
                label_names = [self.get_labels()[int(ind)] for ind in labels[true_conf]]

                self._bbox_normalize(boxes, shape)
                meta_frame = data.get_meta_frames_by_src_name(src_name)[i]
                meta_label = MetaLabel(labels=label_names, confidence=conf)

                meta_frame.add_meta(MetaName.META_BBOX.value, MetaBBox(boxes, meta_label))

    def _bbox_normalize(self, bboxes: torch.tensor, shape: torch.tensor):
        r""" Normalization of bounding box values in the range from 0 to 1.
            :param bboxes: torch.tensor
            :param shape: torch.tensor - image resolution.
            :return:
        """
        bboxes[:, (0, 2)] = bboxes[:, (0, 2)].div(shape[1])
        bboxes[:, (1, 3)] = bboxes[:, (1, 3)].div(shape[0])
        return bboxes


class ModelDetectionDiffLabels(ModelDetection):
    def __init__(self, name: str, model: torch.nn.Module):
        super().__init__(name, model)

    def _add_to_meta(self, data: MetaBatch, preds: list, shape: torch.Tensor, src_name: str, **kwargs):
        r""" Adds bounding boxes to MetaBatch.
            :param data: MetaBatch
            :param preds: A list of floating point values.
            :param shape: torch.tensor - image resolution.
            :param src_name: str - source name
        """
        for i in range(len(preds)):
            boxes = preds[i]['boxes'].cpu()
            label_names = preds[i]['labels']
            conf = preds[i]['scores'].cpu().detach().numpy()
            true_conf = conf > 0.25

            if np.any(true_conf):
                conf = conf[true_conf]
                boxes = boxes[true_conf]
                label_names = np.array(label_names)[true_conf]
                self._bbox_normalize(boxes, shape)
                meta_frame = data.get_meta_frames_by_src_name(src_name)[i]
                meta_label = MetaLabel(labels=label_names, confidence=conf)

                meta_frame.add_meta(MetaName.META_BBOX.value, MetaBBox(boxes, meta_label))


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

    def _to_inference(self, batch: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            probabilities = self._inference(batch)
        return torch.nn.functional.softmax(probabilities[OutputFormat.CONFIDENCE.value]
                                           if isinstance(probabilities, dict) else probabilities, dim=1)

    def _add_to_meta_all(self, meta_batch: MetaBatch, src_data: List, predictions, src_size, *args, **kwargs):
        prob_i = 0
        for i_src_name in range(len(self._source_names)):
            for i in range(src_size[i_src_name]):
                meta_frame = meta_batch.get_meta_frames_by_src_name(self._source_names[i_src_name])[i]
                probability = predictions[prob_i]
                probability = probability[None, :]
                meta_label = MetaLabel(labels=self.get_labels(), confidence=probability)
                meta_frame.add_meta(MetaName.META_LABEL.value, meta_label)
                prob_i += 1


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

    def _to_inference(self, batch: torch.Tensor, *args, **kwargs):
        with torch.no_grad():
            output = self._inference(batch)['out']
        return torch.nn.functional.softmax(output, dim=1)

    def _add_to_meta_all(self, meta_batch: MetaBatch, src_data: List, predictions, src_size, *args, **kwargs):
        prob_i = 0
        for i_src_name in range(len(self._source_names)):
            for i in range(src_size[i_src_name]):
                meta_frame = meta_batch.get_meta_frames_by_src_name(self._source_names[i_src_name])[i]
                normalized_mask = predictions[prob_i]
                mask = torch.zeros(normalized_mask.shape, dtype=torch.bool, device=self.get_device())
                mask[normalized_mask >= self._confidence] = True
                mask = mask[None, :]
                meta_mask = MetaMask(mask, MetaLabel(self.get_labels(), normalized_mask))
                meta_frame.add_meta(MetaName.META_MASK.value, meta_mask)
                prob_i += 1


class ModelDepth(ModelBase):
    r""" Component for stereo models

        :param name: str
                name of component

        :param model: torch.nn.Module
                    stereo model, which returns dictionary with key "out" which contains tensor of shape
                    [N, H, W], where N - batch size, H - mask height, W - mask width and
                    values in the range from 0 to 1.
    """

    def __init__(self, name: str, model: torch.nn.Module, training=False):
        super().__init__(name, model)


    def _to_model_format(self, connected_sources: List[str], data: MetaBatch, device: str, transform,
                         calib=1017.,
                         need_calib=False, **kwargs) -> \
            Tuple[List[Union[Tuple[Any, Any, Any], Tuple[Any, Any]]], List[int]]:
        r""" Returns a list of pairs of transformed frames from the MetaBatch.
            :param connected_sources: list of sources names
            :param data: MetaData
            :param device: str - cuda or cpu
            :param transform: list of transformations from torch.transform
            :return: Tuple[List[Union[Tuple[Any, Any, Any], Tuple[Any, Any]]], List[int]]
        """
        src_data = list()
        if not self.is_train:
            if len(connected_sources) % 2:
                raise ValueError(f'Expected even number of sources, received {len(connected_sources)}')

        def chunk(lst, n) -> List[str]:
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        connected_sources = list(chunk(connected_sources, 2))
        size_frames = [0] * len(connected_sources)

        def clone_data(needed_data):
            cloned_data = needed_data.clone().to(dtype=torch.float, device=device)
            cloned_data = cloned_data.div(255)
            cloned_data = transform(cloned_data)
            return cloned_data

        for i, src_names in enumerate(connected_sources):

            needed_data_left = data.get_frames_by_src_name(src_names[0])
            needed_data_right = data.get_frames_by_src_name(src_names[1])
            size_frames[i] = len(needed_data_right)
            if needed_data_left is None or needed_data_right is None:
                continue

            needed_data_left = clone_data(needed_data_left)
            needed_data_right = clone_data(needed_data_right)
            if need_calib:
                calib = torch.tensor([calib * 0.54]).float().to(dtype=torch.float, device=device)
                src_data.append((needed_data_left, needed_data_right, calib))
            else:
                src_data.append((needed_data_left, needed_data_right))

        return src_data, size_frames

    def _to_inference(self, batch: torch.Tensor, *args, **kwargs):
        output = []
        for pairs in batch:
            imgL, imgR = pairs
            # imgL, imgR, calib = batch
            with torch.no_grad():
                output.append(self._inference((imgL, imgR)))
        return output

    def _add_to_meta_all(self, meta_batch: MetaBatch, src_data: List, predictions, src_size, *args, **kwargs):
        prob_i = 0
        for i_src_name in range(0, len(self._source_names), 2):
            for i in range(src_size[i_src_name // 2]):
                meta_frame = meta_batch.get_meta_frames_by_src_name(self._source_names[i_src_name])[i]
                depth = predictions[prob_i]['depth']
                meta_depth = MetaDepth(depth)
                meta_frame.add_meta(MetaName.META_DEPTH.value, meta_depth)
                prob_i += 1


class DefectsModel(ModelDetection):

    def _to_train(self, batch: torch.Tensor, true_values: MetaBatch):
        self.optimizer.zero_grad()
        key = list(true_values.get_meta_frames_all().keys())[0]
        boxes = [value.get_meta_info(MetaName.META_BBOX.value).get_bbox() for value in
                 true_values.get_meta_frames_by_src_name(key)]
        labels = [value.get_meta_info(MetaName.META_BBOX.value).get_label_info().get_labels() for value in
                  true_values.get_meta_frames_by_src_name(key)]
        out = self._inference(batch, boxes=boxes, labels=labels)
        losses, loss = self.loss_func(out, true_values)
        loss.backward()
        return losses


class LiquidModel(ModelDetection):
    def _to_model_format(self, connected_sources: List[str], data: MetaBatch, device: str, transform, *args,
                         **kwargs) -> Tuple[Tensor, List[Any]]:
        r""" Returns a list of transformed frames from the MetaBatch.
            :param connected_sources:
            :param data: MetaData
            :param device: str - cuda or cpu
            :param transform: list of transformations from torch.transform
            :return: Tuple[Tensor, List[Any]]
        """
        src_data = list()
        for src_name in connected_sources:
            needed_data = data.get_frames_by_src_name(src_name)
            if needed_data is None:
                continue
            cloned_data = needed_data.clone().to(dtype=torch.float, device=device)
            cloned_data = transform(cloned_data)
            src_data.append(cloned_data)
        return self._to_tensor(src_data)
    def _add_to_meta(self, data: MetaBatch, preds: list, shape: torch.Tensor, src_name: str, **kwargs):
        r""" Adds bounding boxes to MetaBatch.
            :param data: MetaBatch
            :param preds: A list of floating point values.
            :param shape: torch.tensor - image resolution.
            :param src_name: str - source name
        """
        for i in range(len(preds)):
            boxes = preds[i].boxes.xywh.cpu()
            labels = preds[i].boxes.cls
            conf = preds[i].boxes.conf.cpu().numpy()
            if conf is None:
                conf = 0
            true_conf = conf > self._confidence

            if np.any(true_conf):
                conf = conf[true_conf]
                boxes = boxes[true_conf]
                label_names = [self.get_labels()[int(ind)] for ind in labels[true_conf]]

                self._bbox_normalize(boxes, shape)
                meta_frame = data.get_meta_frames_by_src_name(src_name)[i]
                meta_label = MetaLabel(labels=label_names, confidence=conf)

                meta_frame.add_meta(MetaName.META_BBOX.value, MetaBBox(boxes, meta_label))



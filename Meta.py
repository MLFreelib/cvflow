import time
from typing import Union, Any, List, Dict

import torch


class MetaLabel:
    r""" Container for storing information about labels and id from tracking.

        :param labels: List[str]
                    list of label names.
        :param confidence: List[float]
                    confidence in the label for each label from labels.
    """

    def __init__(self, labels: List[str], confidence: List[float]):
        self.__labels: List[str] = labels
        self.__object_ids: List[int] = list()
        self.__confidence: List[float] = confidence

    def get_confidence(self) -> torch.tensor:
        r""" Returns tensor of confidences for each label. """
        return self.__confidence

    def set_object_id(self, object_ids: List[int]):
        r""" Sets the ids for each label.
            :param object_ids: List[int].
        """
        if len(object_ids) != len(self.__labels):
            raise ValueError(
                f"Expected number of object IDs {len(self.__labels)}, but received {len(self.__object_ids)}"
            )
        self.__object_ids = object_ids

    def get_labels(self) -> List[str]:
        r""" Returns a list of predicted labels. """
        return self.__labels

    def get_object_ids(self) -> List[int]:
        r""" Returns a list of id for each label. """
        return self.__object_ids


class MetaBBox:
    r""" Container for storing information about bounding boxes.

        :param points: torch.tensor
                    bounding boxes with shape: [N, 4]. Bounding box format: [x_min, y_min, x_max, y_max].
        :param label_info: MetaLabel
                    information about each bounding box.

        :exception TypeError if bbox is not tensor.
        :exception ValueError if the points are of the wrong format.
    """
    def __init__(self, points: torch.tensor, label_info: MetaLabel):

        if not isinstance(points, torch.Tensor):
            raise TypeError(f'Expected type of bbox a torch.Tensor, received {type(points)}')

        if len(points.shape) != 2:
            raise ValueError(f"Expected bbox shape 2, but received {len(points.shape)}")

        if points.shape[1] != 4:
            raise ValueError(f"Expected bbox size 4 (xmin, ymin, xmax, ymax), but received {len(points.shape[1])}")

        self.__points: torch.tensor = points

        labels_count = len(label_info.get_labels())
        if labels_count != points.shape[0]:
            raise ValueError(f"Exptected number of bbox {labels_count}, but received {points.shape[0]}")

        self.__label_info: MetaLabel = label_info

    def get_bbox(self) -> torch.tensor:
        r""" Returns the bounding boxes. """
        return self.__points

    def get_label_info(self) -> MetaLabel:
        r""" Returns a MetaLabel that contains information about the labels for each bounding box. """
        return self.__label_info


class MetaMask:
    r""" Container for storing information about masks

        :param mask: torch.tensor
                    batch of masks.
        :param label_info: MetaLabel
                    information about each mask.

    """
    def __init__(self, mask: torch.tensor, label_info: MetaLabel):

        if len(mask.shape) != 4:
            raise ValueError(f"Expected mask shape 4, but received {len(mask.shape)}")

        self.__mask: torch.tensor = mask

        labels_count = len(label_info.get_labels())
        if labels_count != mask.shape[1]:
            raise ValueError(f"Exptected number of masks {labels_count}, but received {mask.shape[1]}")

        self.__label_info: MetaLabel = label_info

    def get_mask(self) -> torch.tensor:
        r""" Returns the batch of mask. """
        return self.__mask

    def get_label_info(self) -> MetaLabel:
        r""" Returns a MetaLabel that contains information about the labels for each mask. """
        return self.__label_info


class MetaFrame:
    r""" Container for storing a frame information.

        :param source_name: str
                    the name of the source from which the frame was received.
        :param frame: torch.tensor
                    expected shape [3, H, W].
    """
    def __init__(self, source_name: str, frame: torch.tensor):
        self.__frame = None
        self.set_frame(frame)
        self.__labels_info: Union[MetaLabel, None] = None
        self.__mask_info: Union[MetaMask, None] = None
        self.__bbox_info: Union[MetaBBox, None] = None
        self.__custom_meta = dict()
        self.timestamp = time.time()
        self.__source_name = source_name

    def get_label_meta(self) -> Union[MetaLabel, None]:
        r""" Returns a MetaLabel containing information about the label in the frame. """
        return self.__labels_info

    def get_src_name(self) -> str:
        r""" Returns the name of the source from which the frame was received. """
        return self.__source_name

    def set_frame(self, frame: torch.tensor):
        r""" Sets the frame into MetaFrame.
            :param frame: torch.tensor
                        expected shape [3, H, W].
            :exception TypeError if bbox is not tensor.
            :exception ValueError if the points are of the wrong format.
        """
        if not isinstance(frame, torch.Tensor):
            raise TypeError(f'Expected type of frame a torch.Tensor, received {type(frame)}')

        if len(frame.shape) != 3:
            raise ValueError(f"Expected frame shape 3, but received {len(frame.shape)}")

        if frame.shape[0] != 3:
            raise ValueError(f"Expected frame format [3, H, W], but received {frame.shape}")

        self.__frame = frame

    def add_meta(self, meta_name: str, value: Any):
        r""" Add the custom data to the MetaFrame.
            :param meta_name: str
                        name of custom data.
            :param value: Any
                        custom data.

        """
        self.__custom_meta[meta_name] = value

    def get_frame(self) -> torch.Tensor:
        r""" Returns a frame. """
        return self.__frame

    def get_meta_info(self, meta_name: str) -> Any:
        r""" Returns custom data.
            :param meta_name: str
                name of custom data.
        """
        return self.__custom_meta.get(meta_name)

    def set_label_info(self, labels_info: MetaLabel):
        r""" Sets information about the label in the frame.
            :param labels_info: MetaLabel
        """
        if not isinstance(labels_info, MetaLabel):
            raise TypeError(f'Expected type of label a MetaLabel, received {type(labels_info)}')

        self.__labels_info = labels_info

    def get_labels_info(self) -> MetaLabel:
        r""" Returns a MetaLabel that contains information about the labels in the frame. """
        return self.__labels_info

    def set_mask_info(self, mask_info: MetaMask):
        r""" Sets information about predicted masks for this frame.
            :param mask_info: MetaMask.
            :exception TypeError if mask_info is not MetaMask.
        """
        if not isinstance(mask_info, MetaMask):
            raise TypeError(f'Expected type of mask_info a MetaMask, received {type(mask_info)}')

        self.__mask_info = mask_info

    def get_mask_info(self) -> MetaMask:
        r""" Returns the predicted masks for this frame. """
        return self.__mask_info

    def set_bbox_info(self, bbox_info: MetaBBox):
        r""" Sets information about the predicted bounding boxes for this frame.
            :param bbox_info: MetaBBox
        """
        if not isinstance(bbox_info, MetaBBox):
            raise TypeError(f'Expected type of label a MetaBBox, received {type(bbox_info)}')

        self.__bbox_info = bbox_info

    def get_bbox_info(self) -> MetaBBox:
        r""" Returns the predicted bounding boxes for this frame. """
        return self.__bbox_info


class MetaBatch:
    r""" A container for storing a batch of frames.
        :param name: str
                name of batch.
    """
    def __init__(self, name: str):
        self.__name = name
        self.__meta_frames = dict()
        self.__frames = dict()
        self.__source_names = list()

    def add_meta_frame(self, frame: MetaFrame):
        r""" Adds a frame with information about this frame to the batch.
            :param frame: MetaFrame.
            :exception TypeError if frame is not MetaFrame.
        """
        if not isinstance(frame, MetaFrame):
            raise TypeError(f'Expected type of frame a MetaFrame, received {type(frame)}')

        if frame.get_src_name() not in self.__meta_frames.keys():
            self.__meta_frames[frame.get_src_name()] = list()

        self.__meta_frames[frame.get_src_name()].append(frame)

    def add_frames(self, name: str, frames: torch.tensor):
        r""" Adds a frames to the batch.
            :param name: str
                        name of source
            :param frames: torch.tensor.
            :exception TypeError if frames is not tensor.
        """
        if not isinstance(frames, torch.Tensor):
            raise TypeError(f'Expected type of frames a torch.Tensor, received {type(frames)}')

        if name in self.__frames.keys():
            self.__frames[name] = torch.cat((self.__frames[name], torch.unsqueeze(frames, 0)), 0)
        else:
            self.__frames[name] = torch.unsqueeze(frames, 0)

    def get_frames_by_src_name(self, src_name: str) -> Union[torch.tensor, None]:
        r""" Returns frames received from a specific source.
            :param src_name: str
                        name of source.
        """
        return self.__frames[src_name] if src_name in self.__frames.keys() else None

    def get_frames_all(self) -> Dict[str, torch.tensor]:
        r""" Returns all frames.
            :return Dict[str, torch.tensor] where key is name of source and values is frames.
        """
        return self.__frames

    def get_meta_frames_by_src_name(self, src_name: str) -> Union[List[MetaFrame], None]:
        r""" Returns information about frames by the source name. """
        return self.__meta_frames[src_name] if src_name in self.__meta_frames.keys() else None

    def get_meta_frames_all(self) -> Dict[str, List[MetaFrame]]:
        r""" Returns information about all frames in a batch.
            :return Dict[str, MetaFrame] where key is name of source and values is information about frames.
        """
        return self.__meta_frames

    def set_source_names(self, source_names: List[str]):
        r""" Sets the source names.
            :param source_names: List[str]
        """
        if not isinstance(source_names, list):
            raise TypeError(f'Expected type source_names a list of str, received {type(source_names)}')

        self.__source_names = source_names

    def get_source_names(self) -> List[str]:
        r""" Returns the names of all the sources from which the frames were received. """
        return self.__source_names

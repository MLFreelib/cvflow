import time
from enum import Enum
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
                f"Expected number of object IDs {len(self.__labels)}, but received {len(object_ids)}"
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
        self.__label_info = None
        self.__points = None
        self.set_bboxes(points)
        self.set_label_info(label_info)

    def get_bbox(self) -> torch.tensor:
        r""" Returns the bounding boxes. """
        return self.__points

    def set_bboxes(self, points: torch.tensor):
        if not isinstance(points, torch.Tensor):
            raise TypeError(f'Expected type of bbox a torch.Tensor, received {type(points)}')

        if len(points.shape) != 2:
            raise ValueError(f"Expected bbox shape 2, but received {len(points.shape)}")

        if points.shape[1] != 4:
            raise ValueError(f"Expected bbox size 4 (xmin, ymin, xmax, ymax), but received {points.shape[1]}")

        self.__points: torch.tensor = points

    def get_label_info(self) -> MetaLabel:
        r""" Returns a MetaLabel that contains information about the labels for each bounding box. """
        return self.__label_info

    def set_label_info(self, label_info: MetaLabel):
        labels_count = len(label_info.get_labels())
        if labels_count != self.__points.shape[0]:
            raise ValueError(f"Exptected number of bbox {labels_count}, but received {self.__points.shape[0]}")

        self.__label_info: MetaLabel = label_info


class MetaMask:
    r""" Container for storing information about masks
        :param mask: torch.tensor
                    batch of masks.
        :param label_info: MetaLabel
                    information about each mask.
    """

    def __init__(self, mask: torch.tensor, label_info: MetaLabel):
        self.__mask = None
        self.__label_info = None
        self.set_mask(mask)
        self.set_label_info(label_info)

    def get_mask(self) -> torch.tensor:
        r""" Returns the batch of mask. """
        return self.__mask

    def set_mask(self, mask: torch.tensor):
        if len(mask.shape) != 4:
            raise ValueError(f"Expected mask shape 4, but received {len(mask.shape)}")

        self.__mask: torch.tensor = mask

    def get_label_info(self) -> MetaLabel:
        r""" Returns a MetaLabel that contains information about the labels for each mask. """
        return self.__label_info

    def set_label_info(self, label_info: MetaLabel):
        labels_count = len(label_info.get_labels())
        if labels_count != self.__mask.shape[1]:
            raise ValueError(f"Exptected number of labels {labels_count}, but received {self.__mask.shape[1]}")

        self.__label_info: MetaLabel = label_info


class MetaDepth:
    r""" Container for storing information about depth
        :param depth: torch.tensor
                    batch of depth masks.
    """

    def __init__(self, depth: torch.tensor):
        self.__depth = None
        self.set_depth(depth)

    def get_depth(self) -> torch.tensor:
        r""" Returns the batch of depth. """
        return self.__depth

    def set_depth(self, depth: torch.tensor):
        if len(depth.shape) != 3:
            raise ValueError(f"Expected mask shape 3, but received {len(depth.shape)}")

        self.__depth: torch.tensor = depth


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
        self.__meta = dict()
        self.timestamp = time.time()
        self.__source_name = source_name

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

        if frame.shape[0] != 3 and frame.shape[0] != 1:
            raise ValueError(f"Expected frame format [3, H, W] or [1, H, W], but received {frame.shape}")

        self.__frame = frame

    def add_meta(self, meta_name: str, value: Any):
        r""" Add the custom data to the MetaFrame.
            :param meta_name: str
                        name of custom data.
            :param value: Any
                        custom data.
        """
        self.__meta[meta_name] = value

    def get_frame(self) -> torch.Tensor:
        r""" Returns a frame. """
        return self.__frame

    def get_meta_info(self, meta_name: str) -> Any:
        r""" Returns custom data.
            :param meta_name: str
                name of custom data.
        """
        return self.__meta.get(meta_name)


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
        self.__signals = dict()

    def add_signal(self, name: str):
        r""" Adds name to signals.
            :param name: str
                        name of signal.
            :exception TypeError: if name is not str.
        """
        if isinstance(name, str):
            self.__signals[name] = None
        else:
            raise TypeError(f'Expected type of name str, but {type(name)} received.')

    def set_signal(self, name: str, value: Any):
        r""" Sets the value for signals by name.
            :param name: str name of signal
            :param value: Any value of signal.
            :exception TypeError: if name is not str
            :exception ValueError: if the name is missing in the signals.
        """
        if not isinstance(name, str):
            raise TypeError(f'Expected type of name str, but {type(name)} received.')
        if name not in list(self.__signals.keys()):
            raise ValueError(f'Signal {name} not found.')
        self.__signals[name] = value

    def get_signal(self, name: str) -> Union[Any]:
        r""" Returns the value by name.
            :param name: str name of signal
        """
        return self.__signals.get(name)

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


class MetaName(Enum):
    META_BBOX = "meta_bbox",
    META_LABEL = "meta_label",
    META_MASK = "meta_mask",
    META_DEPTH = "meta_depth"

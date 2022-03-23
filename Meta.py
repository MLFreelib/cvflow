import time
from typing import Union, Any, List, Dict

import torch


class MetaLabel:

    def __init__(self, labels: List[str], confidence: List[float]):
        self.__labels: List[str] = labels
        self.__object_ids: List[int] = list()
        self.__confidence: List[float] = confidence

    def get_confidence(self) -> List[float]:
        return self.__confidence

    def set_object_id(self, object_ids: List[int]):
        if len(object_ids) != len(self.__labels):
            raise ValueError(
                f"Expected number of object IDs {len(self.__labels)}, but received {len(self.__object_ids)}"
            )
        self.__object_ids = object_ids

    def get_labels(self) -> List[str]:
        return self.__labels

    def get_object_ids(self) -> List[int]:
        return self.__object_ids


class MetaBBox:
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
        return self.__points

    def get_label_info(self) -> MetaLabel:
        return self.__label_info


class MetaMask:
    def __init__(self, mask: torch.tensor, label_info: MetaLabel):

        if len(mask.shape) != 3:
            raise ValueError(f"Expected bbox shape 3, but received {len(mask.shape)}")

        if mask.shape[1] != 4:
            raise ValueError(f"Expected bbox size 4 (xmin, ymin, xmax, ymax), but received {len(mask.shape[1])}")

        self.__mask: torch.tensor = mask

        labels_count = len(label_info.get_labels())
        if labels_count != mask.shape[0]:
            raise ValueError(f"Exptected number of bbox {labels_count}, but received {mask.shape[0]}")

        self.__label_info: MetaLabel = label_info

    def get_mask(self) -> torch.tensor:
        return self.__mask

    def get_label_info(self) -> MetaLabel:
        return self.__label_info


class MetaFrame:
    def __init__(self, source_name: str, frame: torch.tensor):
        self.__frame = frame
        self.__labels_info: Union[MetaLabel, None] = None
        self.__mask_info: Union[MetaMask, None] = None
        self.__bbox_info: Union[MetaBBox, None] = None
        self.__custom_meta = dict()
        self.timestamp = time.time()
        self.__source_name = source_name

    def get_label_meta(self) -> Union[MetaLabel, None]:
        return self.__labels_info

    def get_src_name(self) -> str:
        return self.__source_name

    def set_frame(self, frame: torch.tensor):
        if not isinstance(frame, torch.Tensor):
            raise TypeError(f'Expected type of frame a torch.Tensor, received {type(frame)}')

        if len(frame.shape) != 3:
            raise ValueError(f"Expected frame shape 3, but received {len(frame.shape)}")

        if frame.shape[0] != 3:
            raise ValueError(f"Expected frame format [3, H, W], but received {frame.shape}")

        self.__frame = frame

    def add_meta(self, meta_name: str, value: Any):
        self.__custom_meta[meta_name] = value

    def get_frame(self) -> torch.Tensor:
        return self.__frame

    def get_meta_info(self, meta_name: str) -> Any:
        return self.__custom_meta.get(meta_name)

    def set_label_info(self, labels_info: MetaLabel):

        if not isinstance(labels_info, MetaLabel):
            raise TypeError(f'Expected type of label a MetaLabel, received {type(labels_info)}')

        self.__labels_info = labels_info

    def get_labels_info(self) -> MetaLabel:
        return self.__labels_info

    def set_mask_info(self, mask_info: MetaMask):

        if not isinstance(mask_info, MetaMask):
            raise TypeError(f'Expected type of label a MetaLabel, received {type(mask_info)}')

        self.__mask_info = mask_info

    def set_bbox_info(self, bbox_info: MetaBBox):

        if not isinstance(bbox_info, MetaBBox):
            raise TypeError(f'Expected type of label a MetaBBox, received {type(bbox_info)}')

        self.__bbox_info = bbox_info

    def get_bbox_info(self) -> MetaBBox:
        return self.__bbox_info


class MetaBatch:
    def __init__(self, name: str):
        self.__name = name
        self.__meta_frames = dict()
        self.__frames = dict()
        self.__source_names = list()

    def add_meta_frame(self, frame: MetaFrame):

        if not isinstance(frame, MetaFrame):
            raise TypeError(f'Expected type of frame a MetaFrame, received {type(frame)}')

        if frame.get_src_name() not in self.__meta_frames.keys():
            self.__meta_frames[frame.get_src_name()] = list()

        self.__meta_frames[frame.get_src_name()].append(frame)

    def add_frames(self, name: str, frames: torch.tensor):

        if not isinstance(frames, torch.Tensor):
            raise TypeError(f'Expected type of frame a torch.Tensor, received {type(frames)}')

        if name in self.__frames.keys():
            self.__frames[name] = torch.cat((self.__frames[name], torch.unsqueeze(frames, 0)), 0)
        else:
            self.__frames[name] = torch.unsqueeze(frames, 0)

    def get_frames_by_src_name(self, src_name: str) -> Union[torch.tensor, None]:
        return self.__frames[src_name] if src_name in self.__frames.keys() else None

    def get_frames_all(self) -> Dict[str, torch.tensor]:
        return self.__frames

    def get_meta_frames_by_src_name(self, src_name: str) -> Union[List[MetaFrame], None]:
        return self.__meta_frames[src_name] if src_name in self.__meta_frames.keys() else None

    def get_meta_frames_all(self) -> Dict[str, List[MetaFrame]]:
        return self.__meta_frames

    def set_source_names(self, source_names: List[str]):

        if not isinstance(source_names, list):
            raise TypeError(f'Expected type source_names a list of str, received {type(source_names)}')

        self.__source_names = source_names

    def get_source_names(self) -> List[str]:
        return self.__source_names

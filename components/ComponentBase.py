from typing import Union

import torch

from Meta import MetaBatch, MetaFrame


class ComponentBase:
    def __init__(self, name: str):
        self._name: str = name
        self._active_status = None
        self._to_component: Union[ComponentBase, None] = None
        self._data_names = list()
        self._source_names = list()
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def add_data_name(self, data_name: str):
        self._data_names.append(data_name)

    def get_data_names(self) -> list:
        return self._data_names

    def connect(self, component):
        self._to_component = component

    def disconnect(self):
        self._to_component = None

    def get_to_component(self):
        return self._to_component

    def get_name(self):
        return self._name

    def get_source_names(self):
        return self._source_names

    def set_source_names(self, source_names: list = None):
        self._sources_name = source_names

    def set_device(self, device: str):
        if device == 'cuda' and torch.cuda.is_available():
            self._device = device
        elif device == 'cpu':
            self._device = device
        else:
            raise TypeError(f'Expected cuda or cpu, actual {device}')

    def get_device(self):
        return self._device

    def do(self, data: Union[MetaBatch, MetaFrame]) -> Union[MetaBatch, MetaFrame]:
        pass

    def start(self):
        pass

    def stop(self):
        pass

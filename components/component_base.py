from typing import Union, List

import torch

from Meta import MetaBatch, MetaFrame


class ComponentBase:
    """ ComponentBase - base class for all components.

        :param name: str
            name of component
    """

    def __init__(self, name: str):
        self._name: str = name
        self._source_names = list()
        self._device = 'cpu'

    def connect(self, component):
        """ Connecting to the next component.
            :param component: ComponentBase
        """
        self._to_component = component

    def disconnect(self):
        r""" Disconnects the following component."""
        self._to_component = None

    def get_to_component(self):
        r""" Returns the component that is next in the pipeline.
            :return to_component: ComponentBase
        """
        return self._to_component

    def get_name(self):
        r""" Returns the name of component """
        return self._name

    def get_source_names(self) -> List[str]:
        r""" Returns the source names. Frames from each input source will be processed in this model. """
        return self._source_names

    def set_source_names(self, source_names: List[str]):
        r""" Sets the source names. Frames from each input source will be processed in this model.
            :param source_names: List[str]
        """
        self._source_names = source_names

    def set_device(self, device: str):
        """
            Sets the type of device on which this component will work.
            :param device: str. Valid values: cuda, cpu.
        """
        if device == 'cuda' and torch.cuda.is_available():
            self._device = device
        elif device == 'cpu':
            self._device = device
        else:
            raise TypeError(f'Expected cuda or cpu, actual {device}')

    def get_device(self) -> str:
        r""" Returns the installed device type on which this component is running.
            :return type of device. Valid values: cuda, cpu.
        """
        return self._device

    def do(self, data: Union[MetaBatch, MetaFrame]) -> Union[MetaBatch, MetaFrame]:
        """ The main method used to implement the functionality of the component.
            :param data: MetaBatch, MetaFrame
                A block of information about personnel.
            :returns: MetaBatch, MetaFrame
                Processed block of information about frames.
        """
        pass

    def start(self):
        """ The method that is executed when compiling the pipeline. """
        pass

    def stop(self):
        """ The method starts when the pipeline stops. This is necessary to delete data and close windows. """
        pass

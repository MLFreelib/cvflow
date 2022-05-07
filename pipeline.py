import time
from typing import List

import cv2
import torch

from components.ComponentBase import ComponentBase
from components.MuxerComponent import MuxerBase, SourceMuxer
from exceptions import InvalidComponentException


class Pipeline:
    r""" A container for building and controlling the pipeline. Allows you to manage components,
        start and stop them.
    """
    def __init__(self):
        self.__device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.__components = list()
        self.__muxer = None

    def add(self, component: ComponentBase):
        r""" Adds a component to the pipeline.
            :param component: ComponentBase.
            :exception InvalidComponentException if the first component in the pipeline is not a MuxerBase.
        """
        if not isinstance(component, MuxerBase) and len(self.__components) == 0:
            raise InvalidComponentException('The first element of the pipeline must be of type MuxerBase')
        self.__components.append(component)

    def set_device(self, device: str):
        r""" Sets the device type. Available types: cpu and cuda. """
        if device == 'cuda' and torch.cuda.is_available():
            self.__device = device
        elif device == 'cpu':
            self.__device = device
        else:
            raise TypeError(f'Expected cuda or cpu, actual {device}')

    def add_all(self, components: List[ComponentBase]):
        r""" Adds a list of components to the pipeline. The data in the pipeline will move through the components
            in the order in which they are in the list.

            :param components: List[ComponentBase].
            :exception TypeError if component is not ComponentBase.
            :exception InvalidComponentException if the first component in the pipeline is not a MuxerBase.
        """
        for component in components:
            if not isinstance(component, ComponentBase):
                self.__components = list()
                raise TypeError(f'Expected {ComponentBase.__name__}, Actual {type(component)}')
            self.add(component)

    def run(self):
        r""" Starts the pipeline. """
        job_time = dict()
        all_time = time.time()
        count = 0

        is_stopped = False
        while not is_stopped:
            data = None
            for i in range(len(self.__components)):
                s_time = time.time()
                data = self.__components[i].do(data)
                e_time = time.time()

                comp_name = self.__components[i].__class__.__name__
                if comp_name not in job_time.keys():
                    job_time[comp_name] = e_time - s_time
                else:
                    job_time[comp_name] = (job_time[comp_name] * count + (
                            e_time - s_time)) / (count + 1)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    is_stopped = True

            count += 1

        all_time = time.time() - all_time
        [print(f'Component: {key}: execution time per iteration {item} sec') for key, item in job_time.items()]
        print(f'FPS: {count / all_time}')

    def compile(self):
        r""" Configures and verifies components. """
        for i in range(0, len(self.__components) - 1):
            if isinstance(self.__components[i], SourceMuxer):
                self.__muxer = self.__components[i]
            self.__components[i].connect(self.__components[i + 1])
            self.__to_device(self.__components[i])

        [component.start() for component in self.__components]

    def close(self):
        r""" Closes each component. """
        [component.stop() for component in self.__components]

    def __to_device(self, component: ComponentBase):
        r""" Sets the device type for the component.
            :param component: ComponentBase.
        """
        component.set_device(device=self.__device)

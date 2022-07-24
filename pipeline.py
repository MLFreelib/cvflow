import time
from enum import Enum
from typing import List

import torch

from Meta import MetaBatch

from common.utils import Logger
from components.component_base import ComponentBase
from components.muxer_component import MuxerBase, SourceMuxer
from exceptions import InvalidComponentException


class Pipeline:
    r""" A container for building and controlling the pipeline. Allows you to manage components,
        start and stop them.
    """

    def __init__(self):
        self.__device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.__components = list()
        self.__signal_names = list()
        self.__logger = self.__create_logger()

    def add(self, component: ComponentBase):
        r""" Adds a component to the pipeline.
            :param component: ComponentBase.
            :exception InvalidComponentException if the first component in the pipeline is not a MuxerBase.
        """
        #if not isinstance(component, MuxerBase) and len(self.__components) == 0:
        #    raise InvalidComponentException('The first element of the pipeline must be of type MuxerBase')
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
            #if not isinstance(component, ComponentBase):
            #    self.__components = list()
            #    raise TypeError(f'Expected {ComponentBase.__name__}, Actual {type(component)}')
            self.add(component)

    def run(self):
        r""" Starts the pipeline. """
        job_time = dict()
        all_time = time.time()
        count = 0
        is_stopped = False
        while not is_stopped:
            data = MetaBatch('pipe_batch')
            data.add_signal(Mode.__name__)
            data.set_signal(Mode.__name__, Mode.PLAY)
            for i in range(len(self.__components)):
                comp_name = self.__components[i].__class__.__name__
                s_time = time.time()
                data = self.__components[int(i)].do(data)
                e_time = time.time()
                if comp_name not in job_time.keys():
                    job_time[comp_name] = e_time - s_time
                else:
                    job_time[comp_name] = (job_time[comp_name] * count + (
                            e_time - s_time)) / (count + 1)
                if data is not None:
                    if data.get_signal(Mode.__name__) == Mode.STOP:
                        is_stopped = True
            count += 1

        all_time = time.time() - all_time
        [self.__logger.write(msg=f'Component: {key}: execution time per iteration {item} sec', lvl='INFO') for key, item
         in job_time.items()]
        self.__logger.write(msg=f'FPS: {count / all_time}', lvl='INFO')

    def compile(self):
        r""" Configures and verifies components. """
        self.__logger.write(msg='Compiling...', lvl='INFO')
        self.__logger.write(msg=f'Device type: {self.__device}.', lvl='INFO')
        for i in range(0, len(self.__components) - 1):
            if isinstance(self.__components[i], SourceMuxer):
                self.__logger.write(msg='SourceMuxer found.', lvl='INFO')
            self.__components[i].connect(self.__components[i + 1])

            cur_comp_name = self.__components[i].__class__.__name__
            next_comp_name = self.__components[i + 1].__class__.__name__
            self.__logger.write(msg=f'{cur_comp_name} connected to {next_comp_name}.', lvl='INFO')
            self.__to_device(self.__components[i])

        [component.start() for component in self.__components]
        self.__logger.write(msg='Compilation completed', lvl='INFO')

    def close(self):
        r""" Closes each component. """
        self.__logger.write(msg='Closing the components...', lvl='INFO')
        [component.stop() for component in self.__components]
        self.__logger.write(msg='Closing the components is complete.', lvl='INFO')

    def add_signals(self, signals: List[str]):
        self.__signal_names = signals

    def __to_device(self, component: ComponentBase):
        r""" Sets the device type for the component.
            :param component: ComponentBase.
        """
        component.set_device(device=self.__device)

    def __create_logger(self):
        logger = Logger()
        logger.add_logger(name='pipeline', options={'handlers': ['consoleHandler'],
                                                    'level': 'INFO'})
        logger.compile_logger(logger_name='pipeline')
        return logger


class Mode(Enum):
    PLAY = 1
    PAUSE = 2
    STOP = 0

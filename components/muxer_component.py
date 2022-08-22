from typing import Union, List

import torch

from Meta import MetaBatch, MetaFrame
from components.component_base import ComponentBase
from components.reader_component import ReaderBase


class MuxerBase(ComponentBase):
    r""" Template for muxer components. """
    pass


class SourceMuxer(MuxerBase):
    r""" Container for components of the ReaderBase type. An important component for the pipeline.
        :param  name: str - name of component
        :param  max_batch_size: int - batch size
    """

    def __init__(self, name: str, max_batch_size=16):
        super().__init__(name)
        self.__current_batch_size = 0
        self.__max_batch_size = max_batch_size
        self.__sources = list()

    def add_source(self, source: ReaderBase):
        r""" Add source to muxer.
            :param source: ReaderBase
            :exception TypeError if source type is not ReaderBase.
        """
        if not isinstance(source, ReaderBase):
            raise TypeError(f'Expected type of source a ReaderBase, received {type(source)}')

        self.__sources.append(source)

    def get_sources(self) -> List[ReaderBase]:
        r""" Returns a list of sources. """
        return self.__sources

    def start(self):
        r""" Starts all input sources. """
        [source.run() for source in self.__sources]

    def do(self, data: Union[MetaBatch, MetaFrame]):
        r""" Reads frames from sources and collects MetaBatch. """
        data.set_source_names([source.get_name() for source in self.__sources])
        source_count = len(self.__sources)

        while True:
            for i in range(source_count):
                frame = self.__sources[i].read()

                meta_frame = self.__to_meta_frame(frame, self.__sources[i].get_name())

                data.add_meta_frame(meta_frame)
                self.__current_batch_size += 1

                if (self.__current_batch_size / source_count) >= self.__max_batch_size:
                    self.__current_batch_size = 0
                    return data

    def __to_meta_frame(self, frame: torch.Tensor, src_name: str) -> MetaFrame:
        r""" Creates MetaFrame from a frame. """
        frame = torch.tensor(frame, device=self.get_device())
        frame = frame.permute(2, 0, 1)
        meta_frame = MetaFrame(source_name=src_name, frame=frame)
        return meta_frame

import cv2
import numpy as np

from components.ComponentBase import ComponentBase
from exceptions import MethodNotOverriddenException


class ReaderBase(ComponentBase):

    def __init__(self, path: str, name: str, framerate: int = 30):
        super().__init__(name)
        self.__is_cuda = False
        self._path = path
        self._last_frame = None
        self._framerate = framerate

    def read(self) -> np.array:
        raise MethodNotOverriddenException('read in the ReaderBase')


class USBCamReader(ReaderBase):
    def __init__(self, src_name: str, name=None, framerate: int = 30):
        super().__init__(path=src_name, name=name, framerate=framerate)
        self.__cap_send = None

    # def run(self):
    #     gstreamer_pipline = f'v4l2src device={self._path} ! video/x-raw,framerate={self._framerate}/1 ! videoscale ! ' \
    #                         f'videoconvert ! appsink'
    #     self.__cap_send = cv2.VideoCapture(gstreamer_pipline, cv2.CAP_GSTREAMER)

    def run(self):
        self.__cap_send = cv2.VideoCapture(self._path)

    def read(self) -> np.array:
        ret, frame = self.__cap_send.read()
        if not ret:
            frame = self._last_frame
        self._last_frame = frame

        return frame


class VideoReader(ReaderBase):

    def __init__(self, path: str, name: str, framerate: int = 30):
        super().__init__(path, name, framerate=framerate)
        self.__cap_send = None
        self.__start_point = 0

    def run(self):
        self.__cap_send = cv2.VideoCapture(self._path)
        if self.__cap_send.isOpened():
            self._framerate = int(self.__cap_send.get(cv2.CAP_PROP_FPS))
        else:
            raise TypeError(f'Could not open the file {self._path}')

    def read(self) -> np.array:
        ret, frame = self.__cap_send.read()
        if not ret:
            return ret, self._last_frame
        self._last_frame = frame

        return frame

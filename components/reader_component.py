import os
import time
from typing import List, Union

import cv2
import numpy as np

from components.component_base import ComponentBase
from exceptions import MethodNotOverriddenException


class ReaderBase(ComponentBase):
    r""" The basic component for reading the video stream.

        :param path: str
                    source location.
        :param name: str
                    name of component
    """

    def __init__(self, path: str, name: str, framerate: int = 30):
        super().__init__(name)
        self._path = path
        self._last_frame = None
        self._framerate = framerate

    def read(self) -> np.array:
        r""" Returns the frame """
        raise MethodNotOverriddenException('read in the ReaderBase')


class CamReader(ReaderBase):
    r""" A component for reading a video stream from a USB camera

        :param device: str
                    location of the USB camera
        :param name: str
                    name of component
    """
    def __init__(self, device: str, name=None, framerate: int = 30):
        super().__init__(path=device, name=name, framerate=framerate)
        try:
            self._path = int(device)
        except ValueError:
            self._path = int(device)
        self.__cap_send = None

    # def run(self):
    #     gstreamer_pipline = f'v4l2src device={self._path} ! video/x-raw,framerate={self._framerate}/1 ! videoscale ! ' \
    #                         f'videoconvert ! appsink'
    #     self.__cap_send = cv2.VideoCapture(gstreamer_pipline, cv2.CAP_GSTREAMER)

    def run(self):
        r""" Creates an instance for video capture. """
        self.__cap_send = cv2.VideoCapture(self._path)

    def read(self) -> np.array:
        r""" Reads frame from usb camera. """
        ret, frame = self.__cap_send.read()
        if not ret:
            frame = self._last_frame
        self._last_frame = frame

        return frame

    def stop(self):
        r""" Clearing memory. """
        self.__cap_send.release()


class VideoReader(ReaderBase):
    r""" A component for reading a video stream from a video file

        :param path: str
                    path to video file
               name: str
                    name of component
    """

    def __init__(self, path: str, name: str, framerate: int = 30):
        super().__init__(path, name, framerate=framerate)
        self.__cap_send = None

    def run(self):
        r""" Creates an instance for video capture. """
        self.__cap_send = cv2.VideoCapture()
        self.__cap_send.open(self._path)
        if self.__cap_send.isOpened():
            self._framerate = int(self.__cap_send.get(cv2.CAP_PROP_FPS))
        else:
            raise TypeError(f'Could not open the file {self._path}')

    def read(self) -> np.array:
        r""" Reads frame from video file. """
        ret, frame = self.__cap_send.read()
        if not ret:
            return ret, self._last_frame
        self._last_frame = frame

        return frame

    def stop(self):
        r""" Clearing memory. """
        self.__cap_send.release()


class ImageReader(ReaderBase):
    r""" Reads the image or images from directory.
        :param: path: str
                path to image or directory with images
        :param: name: str
                name of component
    """
    def __init__(self, path: str, name: str):
        super().__init__(path, name)
        self.__files: Union[List[str], None] = None
        self.__last_file: int = 0
        self.__time_step = 1
        self.__last_iter = None

    def set_time_step(self, step: int = 10):
        r""" Pause between image changes.
            :param step: int
                    seconds
        """
        self.__time_step = step

    def run(self):
        self.__last_iter = time.time()
        if os.path.isfile(self._path):
            self.__files = [self._path]
        elif os.path.isdir(self._path):
            self.__files = os.listdir(self._path)
        else:
            ValueError(f'Expected path to image or directory with images, actual {self._path}')

    def read(self) -> np.array:
        if (time.time() - self.__last_iter) / 1000 < self.__time_step:
            last_file = self.__last_file
        else:
            last_file = max(self.__last_file - 1, 0)
        path = os.path.join(self._path, self.__files[last_file])
        frame = cv2.imread(path)
        self.__last_file = self.__last_file + 1 if self.__last_file + 1 < len(self.__files) else self.__last_file
        self.__last_iter = time.time()
        return frame






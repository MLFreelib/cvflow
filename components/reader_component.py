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
        self.__is_cuda = False
        self._path = path
        self._last_frame = None
        self._framerate = framerate

    def read(self) -> np.array:
        r""" Returns the frame """
        raise MethodNotOverriddenException('read in the ReaderBase')


class USBCamReader(ReaderBase):
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
            pass
        self.__cap_send = None

    # def run(self):
    #     gstreamer_pipline = f'v4l2src device={self._path} ! video/x-raw,framerate={self._framerate}/1 ! videoscale ! ' \
    #                         f'videoconvert ! appsink'
    #     self.__cap_send = cv2.VideoCapture(gstreamer_pipline, cv2.CAP_GSTREAMER)

    def run(self):
        r""" Creates an instance for video capture. """
        self.__cap_send = cv2.VideoCapture(self._path, cv2.CAP_DSHOW)

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
        self.__start_point = 0

    def run(self):
        r""" Creates an instance for video capture. """
        self.__cap_send = cv2.VideoCapture(self._path, cv2.CAP_DSHOW)
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

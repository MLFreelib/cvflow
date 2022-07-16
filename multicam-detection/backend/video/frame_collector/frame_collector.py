import cv2

from exception import EndOfVideoException


class FrameCollector:
    def __init__(self, video):
        self.video = video
        self.video_capture = cv2.VideoCapture(self.video.uri)

    def get_next(self):
        ok, frame = self.video_capture.read()

        if not ok:
            raise EndOfVideoException

        return frame

from abc import ABC, abstractmethod


class Detector(ABC):
    def __init__(self, device):
        self.device = device

    @abstractmethod
    def detect(self, image):
        pass

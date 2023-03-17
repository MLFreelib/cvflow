import torch
import dlib

import numpy as np


class Tracker(torch.nn.Module):
    def __init__(self):
        pass

    def start_track(self, frame, rect):
        tracker = dlib.correlation_tracker()
        tracker.start_track(frame, rect)
        return tracker
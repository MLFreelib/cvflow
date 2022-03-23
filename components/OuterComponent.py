from typing import Union

import cv2

from Meta import MetaBatch, MetaFrame
from components.ComponentBase import ComponentBase


class OuterComponent(ComponentBase):

    def __init__(self, name: str, escape_btn: str = 'q'):
        super().__init__(name)
        self.escape_btn = escape_btn

    def do(self, batch: Union[MetaBatch, MetaFrame]):
        minibatches = batch.get_meta_frames_all()
        key = 'tiler'

        frames = minibatches[key]
        for i in range(len(frames)):
            frame = frames[i].get_frame()
            frame = frame.permute(1, 2, 0).cpu().detach().numpy()

            cv2.waitKey(1)
            cv2.imshow(key, frame)

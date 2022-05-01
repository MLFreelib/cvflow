from typing import Union, List

import cv2

from Meta import MetaBatch, MetaFrame
from components.ComponentBase import ComponentBase


class OuterComponent(ComponentBase):

    def __init__(self, name: str, output: List[str], escape_btn: str = 'q'):
        super().__init__(name)
        self.__escape_btn = escape_btn
        self.__output = output

    def do(self, data: Union[MetaBatch, MetaFrame]):
        full_batch = data.get_meta_frames_all()

        for key in self.__output:
            frames = full_batch[key]
            for i in range(len(frames)):
                frame = frames[i].get_frame()
                frame = frame.permute(1, 2, 0).cpu().detach().numpy()
                cv2.waitKey(1)
                cv2.imshow(key, frame)

    def stop(self):
        cv2.destroyAllWindows()

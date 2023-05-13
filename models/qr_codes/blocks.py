import cv2
import numpy as np
from pyzbar import pyzbar

from Meta import MetaBatch
from models.blocks import Block


class Codes(Block):
    r""" Read QR- and barcodes. """

    def __init__(self, name: str):
        r"""
            :param name: str
                    name of component.
        """
        super().__init__(-1, -1)
        self.__labels = set()  # unique decoding results

    def do(self, data: MetaBatch) -> MetaBatch:
        r""" Read codes from all frames. """
        for src_name in data.get_source_names():
            meta_frames = data.get_meta_frames_by_src_name(src_name)
            for meta_frame in meta_frames:
                frame = meta_frame.get_frame()
                img = frame.detach().cpu().permute(1, 2, 0).numpy()
                frame = self.draw_bboxes_and_write_text(img)
                meta_frame.set_frame(frame)
        return data

    def draw_bboxes_and_write_text(self, img):
        r""" Draw bounds by 4 points and printing qr- or barcode decoding result """
        img = np.ascontiguousarray(img)
        decoded_objects = pyzbar.decode(img)
        for obj in decoded_objects:
            points = obj.polygon
            text = obj.data.decode('UTF-8')
            if len(points) > 4:
                points = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                points = list(map(tuple, np.squeeze(points)))
            n = len(points)
            for i in range(n):
                cv2.line(img, points[i], points[(i + 1) % n], (255, 0, 0), 3)
            cv2.putText(img, text, (points[0].x, points[0].y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1,
                        cv2.LINE_AA)
            if text not in self.__labels:
                print(text)
            self.__labels.add(text)
        return torch.tensor(img, device=self.get_device()).permute(2, 0, 1)

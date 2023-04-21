import threading
from queue import Queue
from typing import Union, List

import cv2
from tensorboardX import SummaryWriter

from pipeline import Mode
from Meta import MetaBatch, MetaFrame
from components.component_base import ComponentBase


class OuterComponent(ComponentBase):
    def __init__(self, name: str):
        super().__init__(name)
        self._source_names = ['tiler']


class DisplayComponent(OuterComponent):
    r""" A component for displaying the pipeline result.

        :param  name: str
                    name of component
        :param  output: List[str]
                    names of sources to display
        :param  escape_btn: str
                    the key to close the window
    """

    def __init__(self, name: str, escape_btn: str = 'q'):
        super().__init__(name)
        self.__escape_btn = escape_btn

    def do(self, data: Union[MetaBatch, MetaFrame]) -> MetaBatch:
        r""" Displays frames from selected sources in the window. """
        full_batch = data.get_meta_frames_all()
        for key in self._source_names:
            frames = full_batch[key]
            for i in range(len(frames)):
                frame = frames[i].get_frame()
                frame = frame.permute(1, 2, 0).cpu().detach().numpy()
                cv2.waitKey(1)
                cv2.imshow(key, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    data.set_signal(Mode.__name__, Mode.STOP)
        return data

    def stop(self):
        r""" Closes windows """
        cv2.destroyAllWindows()


class FileWriterComponent(OuterComponent):
    r""" A component for writing the pipeline result to file.
        :param name: str
                    name of component
        :param file_path: str
                    the path to the file where the result will be saved.
        :param  output: List[str]
                    names of sources to display.
        :param framerate: int
                    the framerate with which the result will be saved.
        :param fourcc: str
                    fourcc code of the codec.

    """

    def __init__(self, name: str, file_path: str, framerate: int = 30, fourcc: str = 'XVID'):
        super().__init__(name)
        self.__path = file_path
        fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.__fcc, self.__fps, self.dt = fourcc, framerate, 1. / framerate
        self._video_writer = None
        self.__queue, self._stop = Queue(maxsize=framerate), False
        self.__thread = threading.Thread(target=self._writer)

    def start(self):
        r""" Starts the thread. """
        self.__thread.start()

    def do(self, data: MetaBatch) -> MetaBatch:
        r""" Puts the frames to the queue."""
        full_batch = data.get_meta_frames_all()
        for key in self._source_names:
            frames = full_batch[key]
            for i in range(len(frames)):
                frame = frames[i].get_frame()
                frame = frame.permute(1, 2, 0).cpu().detach().numpy()
                if self._video_writer is None:
                    res = frame.shape[:2]
                    self._video_writer = cv2.VideoWriter(self.__path, self.__fcc, self.__fps, (res[1], res[0]))
                self.__queue.put(frame)
        return data

    def _writer(self):
        r""" Reads frames from the queue and writes them to the file. """
        last_frame = None
        while True:
            if self._stop:
                break
            while not self.__queue.empty():
                last_frame = self.__queue.get_nowait()
            if last_frame is not None:
                self._video_writer.write(last_frame)

    def stop(self):
        self._stop = True
        self.__thread.join()
        self._video_writer.release()


class TensorBoardOuter(OuterComponent):
    def __init__(self, name: str):
        super().__init__(name)
        self.__writer = SummaryWriter('runs/cvflow_metrics')
        self.__n_iter = 0

    def do(self, data: Union[MetaBatch, MetaFrame]) -> Union[MetaBatch, MetaFrame]:
        r""" Displays frames from selected sources in the window. """
        full_batch = data.get_meta_frames_all()
        for key in self._source_names:
            src_loss = dict()
            frames = full_batch[key]
            frames_count = 0
            loss_names = list()
            for i in range(len(frames)):
                losses = frames[i].get_meta_info("losses")
                loss_names.extend(losses)
                for loss_name in losses:
                    if src_loss.get(loss_name) is None:
                        src_loss[loss_name] = losses[loss_name]
                    else:
                        src_loss[loss_name] = (src_loss[loss_name] * frames_count + losses[loss_name]) / (frames_count + 1)
                frames_count += 1

            for loss_name in loss_names:
                self.__writer.add_scalar(f'{key}_{loss_name}', src_loss[loss_name], self.__n_iter)
        self.__n_iter += 1
        return data

    def stop(self):
        self.__writer.flush()
        self.__writer.close()

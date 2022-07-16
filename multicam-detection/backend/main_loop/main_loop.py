import random
import time

import cv2
import requests
from torch.multiprocessing import get_logger, Process, Manager, Event

from exception import EndOfVideoException
from video.frame_collector import FrameCollector
from video.processor import ProcessorLinker
from video.processor import VideoProcessor


class MainLoop:
    def __init__(self, config):
        self.config = config

        self.session_id = None
        self.video_dict = {}
        self._video_meta = Manager().dict()

        self._video_processor_list = []
        self._video_processor_queue_list = []
        self._video_processor_processes = []

        self._processor_linker = ProcessorLinker(self.config, self._video_meta)

        for device in self.config.available_devices:
            self._video_processor_list.append(VideoProcessor(
                self.config,
                device,
                self._processor_linker.queue,
                self.video_dict
            ))
            self._video_processor_queue_list.append(self._video_processor_list[-1].queue)

            for _ in range(self.config.video_processor_count):
                self._video_processor_processes.append(self._video_processor_list[-1].generate_process())

        self._stop_event = Event()
        self._main_loop_process = Process(
            target=self._loop,
            args=(
                self._video_processor_queue_list,
                self._stop_event,
            )
        )

        self._logger = get_logger()

    def __getstate__(self):
        state = self.__dict__.copy()

        state['_processor_linker'] = None
        state['_video_processor_processes'] = None
        return state

    def start(self):
        self._logger.info("Starting MainLoop")
        self._processor_linker.start()

        for video_processor in self._video_processor_processes:
            video_processor.start()

        self._main_loop_process.start()
        self._logger.info("MainLoop started")

    def stop(self):
        self._logger.info("Stopping MainLoop")
        self._stop_event.set()

        if self._main_loop_process.is_alive():
            self._main_loop_process.join()

    def kill(self):
        self._logger.info("Killing MainLoop")
        self._main_loop_process.kill()

        for video_processor in self._video_processor_processes:
            video_processor.kill()

        self._processor_linker.kill()
        self._logger.info("MainLoop killed")

    def append_video(self, video):
        video_id = video.video_id
        self.video_dict[video_id] = video

        capture = cv2.VideoCapture(video.uri)

        self._video_meta[video_id] = {
            "fps": capture.get(cv2.CAP_PROP_FPS),
            "width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
        }

        _, frame = capture.read()
        self._send_screenshot(video_id, self.session_id, frame)

        capture.release()

    def _send_screenshot(self, video_id, session_id, frame):
        self._logger.debug(
            "Sending screenshot of video with id '{}' for session with id '{}'".format(video_id, session_id))

        method = "http"
        if self.config.secure:
            method = "https"
        requests.post("{}://{}:{}/api/v{}/video/screenshot?id={}&session={}".format(
            method,
            self.config.host,
            self.config.port,
            self.config.api_version,
            video_id,
            session_id
        ), data=cv2.imencode(".jpg", frame)[1].tobytes())

    def _loop(self, video_processor_queue_list, stop_event):
        self._logger.info("MainLoop started")
        self._logger.info("MainLoop start collecting frames")

        frame_collector_dict = {}
        for video_id in self.video_dict:
            video = self.video_dict[video_id]

            frame_collector_dict[video_id] = FrameCollector(video)

        batches = {}

        iteration_id = 0

        last_time = time.time()
        while True:
            if stop_event.is_set():
                break

            progress = False

            current_time = time.time()
            if current_time - last_time < 1.0 / self.config.fps:
                continue

            self._logger.debug("MainLoop collecting frames with iteration id '{}'".format(iteration_id))
            last_time = current_time

            for video_id in frame_collector_dict:

                if video_id not in batches:
                    batches[video_id] = []

                try:
                    frame = frame_collector_dict[video_id].get_next()

                    batches[video_id].append((iteration_id, frame))

                    progress = True
                except EndOfVideoException:
                    batches[video_id].append((iteration_id, None))

            if not progress:
                break

            iteration_id += 1

            if iteration_id % self.config.stride_between_detection == 0:
                for video_id in batches:
                    video_processor_queue_list[random.randint(0, len(video_processor_queue_list) - 1)] \
                        .put((video_id, batches[video_id]))

                    batches[video_id] = []

        self._logger.info("MainLoop done collecting frames")

        for video_processor_queue in video_processor_queue_list:
            video_processor_queue.put(None)

        self._logger.info("MainLoop stopped")

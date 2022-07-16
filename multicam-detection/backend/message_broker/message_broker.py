import asyncio
import json
import os
import time
from threading import Event, Thread

import numpy as np
import requests
import websockets
from torch.multiprocessing import get_logger

from camera import Camera
from camera.calibration.calibration import Calibration
from main_loop import MainLoop
from video.video import Video


class MessageBroker:
    def __init__(self, config):
        self.config = config
        self._logger = get_logger()
        self.readers = []
        self._kill_thread_event = Event()

        self._main_loop = MainLoop(config)

        self._video_download_processes = []

    def start(self):
        self._logger.info("Starting MessageBroker")

        asyncio.run(self._work())
        self._logger.info("MessageBroker started")

    def kill(self):
        self._logger.info("Killing MessageBroker")
        self._kill_thread_event.set()
        self._logger.info("MessageBroker killed")

    async def _work(self):
        method = "ws"
        if self.config.secure:
            method = "wss"
        async with websockets.connect(
                "{}://{}:{}/backend/websocket".format(method, self.config.host, self.config.port)
        ) as websocket:
            self._logger.info("Establish websocket connection with frontend")

            while True:
                if self._kill_thread_event.is_set():
                    break

                self._logger.info("Awaiting command from frontend")

                response_data = await websocket.recv()
                response_json = json.loads(response_data)
                self._proceed_command(response_json)

    def _download_video(self, video_id, path):
        self._logger.info("Start downloading video with id '{}'".format(video_id))

        while True:
            method = "http"
            if self.config.secure:
                method = "https"
            response = requests.get("{}://{}:{}/api/v{}/{}".format(
                method,
                self.config.host,
                self.config.port,
                self.config.api_version,
                path
            ))

            if response.status_code != 200:
                self._logger.debug("Failed to download video with id {}. Retry".format(video_id))
                time.sleep(1)
            else:
                break

        video_dir_path = "{}/{}".format(
            os.path.dirname(os.path.abspath(__file__)),
            self.config.save_file_video_dir
        )

        if not os.path.exists(video_dir_path):
            os.makedirs(video_dir_path, exist_ok=True)

        video_file_path = "{}/file-video-vid#{}-sid#{}".format(
            video_dir_path,
            video_id,
            self._main_loop.session_id
        )

        with open(video_file_path, "wb") as video_file:
            video_file.write(response.content)
            video_file.flush()


        self._main_loop.append_video(
            Video(
                video_id,
                self._main_loop.session_id,
                video_file_path,
                "file",
                Camera()
            )
        )

        self._logger.info("End downloading vido with id '{}'".format(video_id))

    def _proceed_command(self, response):
        command = response["command"]
        self._logger.info("Processing command '{}', received from frontend".format(command))

        if command == "START_SESSION":
            session_id = response["sessionId"]

            if self._main_loop.session_id is None:
                self._main_loop.session_id = session_id
            else:
                self._logger.warning("Incorrect command START_SESSION: session is already set")
        elif command == "STOP_SESSION":
            session_id = response["sessionId"]

            if self._main_loop.session_id == session_id:
                self._main_loop.stop()

                del self._main_loop
                self._main_loop = MainLoop(self.config)
            else:
                self._logger.warning("Incorrect session for command STOP_SESSION")
        elif command == "APPEND_VIDEO":
            video_id = response["videoId"]
            session_id = response["sessionId"]
            uri = response["uri"]
            streaming_type = response["streamingType"]

            if self._main_loop.session_id == session_id:
                self._logger.info("Appending video with uri '{}'".format(uri))

                if streaming_type == "file":
                    self._video_download_processes.append(Thread(
                        target=self._download_video,
                        args=(video_id, uri,)
                    ))

                    self._video_download_processes[-1].start()
                else:
                    self._main_loop.append_video(
                        Video(
                            video_id,
                            self._main_loop.session_id,
                            uri,
                            streaming_type,
                            Camera()
                        )
                    )
            else:
                self._logger.warning("Incorrect session for command APPEND_VIDEO")
        elif command == "START_STREAMING":
            session_id = response["sessionId"]

            if self._main_loop.session_id == session_id:
                for video_download_process in self._video_download_processes:
                    video_download_process.join()

                self._main_loop.start()
            else:
                self._logger.warning("Incorrect session for command START_STREAMING")
        elif command == "SET_CALIBRATION":
            video_id = response["videoId"]
            session_id = response["sessionId"]
            points = response["calibrationPointList"]

            if self._main_loop.session_id == session_id:
                screen_points = np.array([])
                world_points = np.array([])
                for point in points:
                    x_screen = point["xScreen"]
                    y_screen = point["yScreen"]

                    screen_points = np.append(screen_points, [x_screen, y_screen])

                    x_word = point["xWorld"]
                    y_word = point["yWorld"]
                    z_word = point["zWorld"]

                    world_points = np.append(world_points, [x_word, y_word, z_word])

                screen_points = screen_points.reshape((6, 2))
                world_points = world_points.reshape((6, 3))

                calibration = Calibration(screen_points, world_points)
                self._logger.debug("Calibration set to video with id '{}' with matrix\n{}".format(
                    video_id,
                    calibration.matrix
                ))
                self._main_loop.video_dict[video_id].camera.calibration = calibration
            else:
                self._logger.warning("Incorrect session for command SET_CALIBRATION")
        else:
            self._logger.warning("Unknown command '{}', received from frontend".format(command))

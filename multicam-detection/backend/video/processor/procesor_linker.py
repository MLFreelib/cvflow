import base64
import json
import subprocess
import tempfile
from threading import Thread

import cv2
import ffmpeg
import requests
from torch.multiprocessing import Process, get_logger, Queue

from backend projector import Projector


class ProcessorLinker:
    def __init__(self, config, video_meta):
        self.config = config
        self.video_meta = video_meta

        self.queue = Queue()
        self._linker_process = Process(target=self._link, args=(self.queue, self.video_meta,))

        self._projector = Projector(self.config)

        self._logger = get_logger()

    def start(self):
        self._logger.info("Starting ProcessLinker")
        self._linker_process.start()
        self._logger.info("ProcessLinker started")

    def join(self):
        self._logger.info("Joining ProcessLinker")
        self._linker_process.kill()
        self._logger.info("ProcessLinker joined")

    def kill(self):
        self._logger.info("Killing ProcessLinker")
        self._linker_process.kill()
        self._logger.info("ProcessLinker killed")

    def _convert_and_send(self, sequence_id, batches):
        self._logger.debug("Prepare to send sequence with id '{}'".format(sequence_id))

        fragments_json = []
        frame_id_video_id_dict = {}

        for video_id in batches:
            video_batch = batches[video_id]

            width = int(self.video_meta[video_id]["width"])
            height = int(self.video_meta[video_id]["height"])

            with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_file_mp4:
                writer = cv2.VideoWriter(
                    tmp_file_mp4.name,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.config.fps,
                    (width, height)
                )

                for frame_id, (frame, projection_class_idx_array) in enumerate(video_batch):
                    if frame is not None:
                        writer.write(frame)

                    if frame_id not in frame_id_video_id_dict:
                        frame_id_video_id_dict[frame_id] = {}

                    frame_id_video_id_dict[frame_id][video_id] = projection_class_idx_array

                writer.release()

                with tempfile.NamedTemporaryFile(suffix=".ts") as tmp_file_ts:
                    ffmpeg \
                        .input(tmp_file_mp4.name) \
                        .output(tmp_file_ts.name, vcodec='libx264', acodec='aac', audio_bitrate='160K',
                                vbsf='h264_mp4toannexb', format='mpegts',
                                muxdelay=0,
                                output_ts_offset=str(
                                    float(sequence_id * self.config.stride_between_send) / self.config.fps)) \
                        .run(capture_stdout=True, capture_stderr=True, quiet=True, overwrite_output=True)

                    duration_format = subprocess.check_output([
                        'ffprobe', '-i', tmp_file_ts.name, '-show_entries', 'format=duration', '-v', 'quiet', '-of',
                        'json'
                    ]).decode('utf8')
                    duration_format_json = json.loads(duration_format)
                    duration = duration_format_json["format"]["duration"]

                    data = tmp_file_ts.read()

                    fragments_json.append({
                        "duration": duration,
                        "videoId": video_id,
                        "data": base64.b64encode(data).decode('utf-8')
                    })

                    tmp_file_ts.close()

                tmp_file_mp4.close()

        method = "http"
        if self.config.secure:
            method = "https"
        response = requests.post("{}://{}:{}/api/v{}/session/batch/{}".format(
            method,
            self.config.host,
            self.config.port,
            self.config.api_version,
            sequence_id + 1,
        ), json={
            "fragments": fragments_json,
            "projections": self._projector.get_next_projection_batch(frame_id_video_id_dict)
        })

        self._logger.debug("Sent batch '{}' to frontend with response '{}".format(sequence_id, response))

    # noinspection PyUnboundLocalVariable,DuplicatedCode
    def _link(self, queue, video_meta):
        processed_dict = {}

        next_sequence_id = 0
        next_iteration_id = 0

        batches = {}
        append_cnt = 0

        done_cnt = 0

        while True:
            data = queue.get()

            if data is None:
                done_cnt += 1

                if done_cnt == len(self.config.available_devices) * self.config.video_processor_count:
                    break

                continue

            video_id, iteration_id, frame, projection_class_idx_array = data

            if iteration_id not in processed_dict:
                processed_dict[iteration_id] = []

            processed_dict[iteration_id].append((video_id, frame, projection_class_idx_array))

            while next_iteration_id in processed_dict and len(processed_dict[next_iteration_id]) == len(video_meta):
                self._logger.debug("Start linking iteration id '{}'".format(iteration_id))

                for video_id, frame, projection_class_idx_array in processed_dict[next_iteration_id]:
                    if video_id not in batches:
                        batches[video_id] = []

                    batches[video_id].append((frame, projection_class_idx_array))
                    append_cnt += 1

                del processed_dict[next_iteration_id]

                if append_cnt == self.config.stride_between_send * len(video_meta):
                    Thread(target=self._convert_and_send, args=(next_sequence_id, batches,)).start()
                    next_sequence_id += 1

                    batches = {}

                    append_cnt = 0

                next_iteration_id += 1

        while next_iteration_id in processed_dict:
            self._logger.debug("Start linking iteration id '{}'".format(next_iteration_id))

            for video_id, frame, projection_class_idx_array in processed_dict[next_iteration_id]:
                if video_id not in batches:
                    batches[video_id] = []

                batches[video_id].append((frame, projection_class_idx_array))
                append_cnt += 1

            del processed_dict[next_iteration_id]

            if append_cnt == self.config.stride_between_send * len(video_meta):
                Thread(target=self._convert_and_send, args=(next_sequence_id, batches,)).start()
                next_sequence_id += 1

                batches = {}

                append_cnt = 0

            next_iteration_id += 1

        method = "http"
        if self.config.secure:
            method = "https"
        requests.post("{}://{}:{}/api/v{}/session/stop".format(
            method,
            self.config.host,
            self.config.port,
            self.config.api_version
        ))

        self._logger.info("Process linker stopped")

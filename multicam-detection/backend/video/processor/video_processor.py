import cv2
import dlib
import numpy as np
from PIL import Image
from detectron2 import model_zoo
from torch.multiprocessing import Queue, Process, get_logger

from bbox_expander import BboxExpander
from detector import YoloDetector
from detector.detectron2_detector import Detectron2Detector
from detector.object_class.object_class_type import ObjectClassType


class VideoProcessor:
    def __init__(self,
                 config,
                 device,
                 processor_linker_queue,
                 video_dict
                 ):
        self.config = config
        self.device = device
        self.processor_linker_queue = processor_linker_queue
        self.video_dict = video_dict

        self.queue = Queue()

        if self.config.detector_type.startswith("yolov5"):
            self._detector = YoloDetector(
                device,
                self.config.detector_type,
                self.config.detector_weight_path
            )
        else:
            weights = self.config.detector_weight_path
            if weights is None:
                weights = model_zoo.get_checkpoint_url("{}.yaml".format(self.config.detector_type))
            self._detector = Detectron2Detector(
                device,
                "{}.yaml".format(self.config.detector_type),
                weights
            )
        if self.config.bbox_expander_type is None:
            self._bbox_expander = None
        else:
            self._bbox_expander = BboxExpander(device, self.config.bbox_expander_weight_path)

        self._logger = get_logger()

    def generate_process(self):
        self._logger.info("Generating VideoProcessor")

        process = Process(
            target=self._process_loop,
            args=(
                self.queue,
                self.processor_linker_queue,
            )
        )

        self._logger.info("VideoProcessor generated")

        return process

    def _process_loop(self,
                      queue,
                      processor_linker_queue):
        self._logger.info("VideoProcessor started")

        while True:
            data = queue.get()

            if data is None:
                break

            video_id, batch = data

            self._process_batch(
                processor_linker_queue,
                video_id,
                batch
            )

        self.processor_linker_queue.put(None)

        self._logger.info("VideoProcessor stopped")

    # noinspection DuplicatedCode
    def _process_batch(self, processor_linker_queue, video_id, batch):
        expand_list = []
        tracker_class_list = []

        for frame_id, (iteration_id, frame) in enumerate(batch):
            self._logger.debug("Start processing frame with iteration id '{}' for video id '{}' and frame id '{}'"
                               .format(iteration_id, video_id, frame_id))
            projection_class_idx_array = np.array([]).reshape((-1, 5))

            if frame is None:
                self._logger.debug("Processed dummy frame with iteration id '{}' for video id '{}'"
                                   .format(iteration_id, video_id))
                processor_linker_queue.put((video_id, iteration_id, None, None))
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame_id == 0:
                self._logger.debug(
                    "Starting detection for frame with iteration id '{}' and frame if '{}' for video id '{}'"
                        .format(iteration_id, frame_id, video_id)
                )
                bboxes, classes, scores = self._detector.detect(frame)  # save to meta
                self._logger.debug(
                    "Done detection for frame with iteration id '{}' and frame id '{}' for video id '{}'"
                        .format(iteration_id, frame_id, video_id)
                )

                for idx, (obj_bbox, obj_class, obj_score) in enumerate(zip(bboxes, classes, scores)):
                    if obj_class != ObjectClassType.PERSON.value and obj_class != ObjectClassType.CAR.value:
                        continue

                    if obj_score < self.config.detector_threshold:
                        continue

                    tracker_class_list.append((idx, dlib.correlation_tracker(), obj_class))
                    tracker_class_list[-1][1].start_track(
                        rgb_frame,
                        dlib.rectangle(
                            int(obj_bbox[0]),
                            int(obj_bbox[1]),
                            int(obj_bbox[2]),
                            int(obj_bbox[3])
                        )
                    )

                    croped_image = rgb_frame[
                                   int(obj_bbox[1]):int(obj_bbox[3]),
                                   int(obj_bbox[0]):int(obj_bbox[2])
                                   ]

                    if self._bbox_expander is not None:
                        expand = self._bbox_expander.expand(Image.fromarray(croped_image))
                        expand_bbox = BboxExpander.apply_expand(obj_bbox, expand)

                        final_bbox = expand_bbox

                        expand_list.append(expand)
                    else:
                        final_bbox = obj_bbox

                        expand_list.append(None)

                    mean_height = 0.0
                    if obj_class == ObjectClassType.PERSON.value:
                        mean_height = self.config.person_mean_height
                    if obj_class == ObjectClassType.CAR.value:
                        mean_height = self.config.car_mean_height

                    if self.video_dict[video_id].camera.calibration is not None:
                        p1 = self.video_dict[video_id].camera.calibration.project_2d_to_3d(
                            np.array([(final_bbox[2] + final_bbox[0]) / 2.0, final_bbox[3]]),
                            Z=0
                        )
                        p2 = self.video_dict[video_id].camera.calibration.project_2d_to_3d(
                            np.array([(final_bbox[2] + final_bbox[0]) / 2.0, final_bbox[1]]),
                            Z = mean_height
                        )
                        p3 = np.array([p2[0], p2[1], 0])
                        projection = p1 + (p3 - p1) / 2.0
                        projection_class_idx_array = np.append(
                            projection_class_idx_array,
                            np.append(projection, [obj_class, idx]).reshape((1, 5)),
                            axis=0
                        )
            else:
                for (idx, tracker, obj_class), expand in zip(tracker_class_list, expand_list):
                    tracker.update(rgb_frame)
                    tracker_position = tracker.get_position()

                    bbox = [
                        tracker_position.left(),
                        tracker_position.top(),
                        tracker_position.right(),
                        tracker_position.bottom()
                    ]

                    if self._bbox_expander:
                        expand_bbox = BboxExpander.apply_expand(bbox, expand)

                        final_bbox = expand_bbox
                    else:
                        final_bbox = bbox

                    mean_height = 0.0
                    if obj_class == ObjectClassType.PERSON.value:
                        mean_height = self.config.person_mean_height
                    if obj_class == ObjectClassType.CAR.value:
                        mean_height = self.config.car_mean_height

                    if self.video_dict[video_id].camera.calibration is not None:
                        p1 = self.video_dict[video_id].camera.calibration.project_2d_to_3d(
                            np.array([(final_bbox[2] + final_bbox[0]) / 2.0, final_bbox[3]]),
                            Z=0
                        )
                        p2 = self.video_dict[video_id].camera.calibration.project_2d_to_3d(
                            np.array([(final_bbox[2] + final_bbox[0]) / 2.0, final_bbox[1]]),
                            Z=mean_height
                        )
                        p3 = np.array([p2[0], p2[1], 0])
                        projection = p1 + (p3 - p1) / 2.0
                        projection_class_idx_array = np.append(
                            projection_class_idx_array,
                            np.append(projection, [obj_class, idx]).reshape((1, 5)),
                            axis=0
                        )
            processor_linker_queue.put((video_id, iteration_id, frame, projection_class_idx_array))
            self._logger.debug("Done processing frames with iteration id '{}' for video id '{}'"
                               .format(iteration_id, video_id))

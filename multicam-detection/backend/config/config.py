from torch.multiprocessing import get_logger


class Config:
    def __init__(self):
        self._logger = get_logger()

        self.host = "localhost"

        self.port = 4000

        self.secure = False

        self.api_version = "1"

        self.stride_between_detection = 10

        self.stride_between_send = 60

        self.available_devices = ["cpu"]

        self.detector_type = None

        self.detector_weight_path = None

        self.detector_threshold = 0.65

        self.bbox_expander_type = None

        self.bbox_expander_weight_path = "data/bbox_expander/weight/model_final.pth"

        self.fps = 30

        self.person_radius = 0.2

        self.person_mean_height = 1.65

        self.person_mean_velocity_in_dist_per_sec = 0.7

        self.car_radius = 0.75

        self.car_mean_height = 2.15

        self.car_mean_velocity_in_dist_per_sec = 1.8

        self.decay_point_time_sec = 2

        self.save_file_video_dir = "data/video"

        self.video_processor_count = 1

        self.logger_type = "INFO"

    def assert_correct(self):
        if self.stride_between_send % self.stride_between_detection != 0:
            self._logger.error("'stride_between_send' must be dividable by 'stride_between_detection'")
            raise Exception()

    def person_mean_distance_per_frame(self):
        return self.person_mean_velocity_in_dist_per_sec / self.fps

    def car_mean_distance_per_frame(self):
        return self.car_mean_velocity_in_dist_per_sec / self.fps

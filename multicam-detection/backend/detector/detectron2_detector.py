from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from detector import Detector

classes_names = ["person"]


class Detectron2Detector(Detector):
    def __init__(self,
                 device,
                 model_name,
                 model_weights):
        super().__init__(device)
        detector_config = get_cfg()
        detector_config.merge_from_file(model_zoo.get_config_file(model_name))
        detector_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        detector_config.MODEL.WEIGHTS = model_weights
        detector_config.MODEL.DEVICE = device
        self._predictor = DefaultPredictor(detector_config)

    def detect(self, image):
        output = self._predictor(image)

        classes = output["instances"].pred_classes.cpu()
        bboxes = output["instances"].pred_boxes.tensor.cpu()
        scores = output["instances"].scores.cpu()

        return bboxes, classes, scores

import torch

from detector import Detector


class YoloDetector(Detector):
    def __init__(self, device, yolo_version, model_weight_path=None):
        super().__init__(device)

        self.yolo_version = yolo_version
        self.model = torch.hub.load('ultralytics/yolov5', self.yolo_version, pretrained=True)
        if model_weight_path is not None:
            self.model.load_state_dict(torch.load(model_weight_path, map_location=device))

        self.model.to(device)
        self.model.share_memory()
        self.model.eval()

    def detect(self, image):
        output = self.model(image)
        df = output.pandas().xyxy[0]

        classes = df.loc[:, "class"].to_numpy()
        scores = df.loc[:, "confidence"].to_numpy()
        bboxes = df.iloc[:, 0:4].to_numpy()

        return bboxes, classes, scores

import torch
import sys

sys.path.append('yolo_models/yolov5')
from models.yolov5.models.common import DetectMultiBackend, AutoShape

class YOLO(torch.nn.Module):
    def __init__(self, clf_spec=None):
        super().__init__()
        self.cls_model = None
        dmb = DetectMultiBackend(weights='yolov5l.pt', device='cpu')
        self.model = AutoShape(dmb)
        self.boxes = []

    def forward(self, x):
        res = self.model(x).xyxy[0]
        self.boxes = res[..., :4]
        if self.cls_model:
            labels = torch.tensor(self.cls_model.one_image_result(nx, res[..., :4]))
        else:
            labels = res[..., 5]
        out = [{'boxes': res[..., :4],
                'labels': labels,
                'scores': res[..., 4]}, ]
        return out
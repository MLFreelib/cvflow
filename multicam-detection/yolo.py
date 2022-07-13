import torch


class YOLO(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
        self.boxes = []

    def forward(self, x):
        nx = x.cpu().detach().numpy() * 255
        res = self.model(nx[0]).xyxy[0]
        self.boxes = res[..., :4]
        out = [{'boxes': res[..., :4],
                'labels': res[..., 5],
                'scores': res[..., 4]}, ]
        return out

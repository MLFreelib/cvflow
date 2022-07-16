import torch
from backend.projector.projector import Projector
from config.config import Config
from set_stream import get_config

class YOLO(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
        self.boxes = []
        config = get_config()
        self.projector = Projector(Config())

    def forward(self, x):
        nx = x.cpu().detach().numpy() * 255
        #res = self.model(nx[0]).xyxy
        #print(nx.shape)
        #res = self.model(nx[0]).xyxy[0]
        #self.boxes = res[..., :4]
        out = []
        #res = self.model(nx)

        for i in nx:
            res = self.model(i).xyxy[0]
            out.append({'boxes': res[..., :4],
                        'labels': res[..., 5],
                        'scores': res[..., 4]})

        #out = [{'boxes': res[..., :4],
        #        'labels': res[..., 5],
        #        'scores': res[..., 4]}, ]
        return out

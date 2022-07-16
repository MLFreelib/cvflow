import torch
from torchvision import transforms

from bbox_expander import BboxExpandNet


class BboxExpander:
    def __init__(self, device, model_weight_path):
        self.device = torch.device(device)

        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.model_weight_path = model_weight_path

        self.model = BboxExpandNet()
        self.model.to(self.device)
        self.model.share_memory()
        self.model.load_state_dict(torch.load(self.model_weight_path, map_location=self.device))
        self.model.eval()

    def expand(self, image):
        expand = self.model(self.data_transforms(image).unsqueeze(0).to(self.device)).tolist()[0]

        return expand

    @staticmethod
    def apply_expand(bbox, expand):
        x = bbox[0]
        y = bbox[1]
        w = bbox[2] - x
        h = bbox[3] - y
        x += w / 2.0
        y += h / 2.0

        nw = w / expand[0]
        nh = h / expand[1]
        nx = expand[2] * (nw / 2.0) + x
        ny = expand[3] * (nh / 2.0) + y

        return [nx - nw / 2.0, ny - nh / 2.0, nx + nw / 2.0, ny + nh / 2.0]

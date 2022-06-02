import torch
from vehicles_classifier import SuperModel, classes


class YOLO(torch.nn.Module):
    def __init__(self, clf_spec=None):
        super().__init__()
        if clf_spec == 'vehicles':
            root_path = 'Practice/'
            cls_cfgs = (root_path + 'configs/classifiers/resnext/resnext101-32x8d_8xb32_in1k.py',
                        root_path + 'configs/classifiers/swin_transformer/swin-base_16xb64_in1k.py',
                        root_path + 'configs/classifiers/convnext/convnext-xlarge_64xb64_in1k.py',
                        root_path + 'configs/classifiers/seresnet/seresnet101.py',
                        )
            cls_num = 1
            cls_model_name = cls_cfgs[cls_num].split('/')[-2]
            chkpnt = root_path + 'logs/classifiers/' + cls_model_name + '/latest.pth'
            supermodel = SuperModel(classes, 'CPU')
            supermodel._init_cls_inference(cls_cfgs[cls_num], chkpnt)
            self.cls_model = supermodel
        else:
            self.cls_model = None
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
        self.boxes = []

    def forward(self, x):
        nx = x.cpu().detach().numpy() * 255
        res = self.model(nx[0]).xyxy[0]
        self.boxes = res[..., :4]
        if self.cls_model:
            labels = torch.tensor(self.cls_model.one_image_result(nx, res[..., :4]))
        else:
            labels = res[..., 5]
        out = [{'boxes': res[..., :4],
                'labels': labels,
                'scores': res[..., 4]}, ]
        return out

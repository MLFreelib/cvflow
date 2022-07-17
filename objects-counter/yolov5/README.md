<div align="center">
<p>
   <a align="left" href="https://ultralytics.com/yolov5" target="_blank">
   <img width="85" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/splash.jpg"></a>
</p>

English | [简体中文](.github/README_cn.md)
   
# Using YOLO
```
from yolov5.models.common import DetectMultiBackend, AutoShape
dmb = DetectMultiBackend(weights='yolov5l.pt', device='cpu')
model = AutoShape(dmb)

def forward(x:torch.Tensor):
    res = model(x).xyxy[0] #  for one image batch
    boxes = res[..., :4]
    scores = res[..., 4]
    labels = res[..., 5]
```

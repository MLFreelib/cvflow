# Running config creator
```
python3 config_gui.py --video <path_to_video> -n <config_file_name>
```

# Running counter
```
python3 run.py --videofile <path_to_video> -f <path_to_fonts> -c <path_to_config_file>
```

# Using yolo
## Install package
```

```


## Run detection
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

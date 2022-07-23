# Objects flows counter
Module for counting objects in the flow. Consists of YOLOv5 detector and correlational tracker.
## Metrics
|Model|IoU|F1|Precision|Recall|
|-----|---|--|---------|------|
|YOLOv5 Large|0.5|0.7716|0.8977|0.6766|
|YOLOv5 Large|0.75|...|...|...|
|YOLOv5 Large|0.9|...|...|...|

## Running config creator
```
python3 config_gui.py --video <path_to_video> -n <config_file_name>
```

## Running counter
running parameters:
- --usbcam
- --videofile
- -c, --config - path to the configuration file
- -t, --tracking_frames - number of frames without detection
- -w, --width - video frame width
- -f, --font
```
python3 run.py --videofile <path_to_video> -f <path_to_fonts> -c <path_to_config_file>
```

# Using YOLO
Read [yolov5/README.md](https://github.com/MLFreelib/cvflow/blob/objects-counter/objects-counter/yolov5/README.md)

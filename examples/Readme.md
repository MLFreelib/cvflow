# Examples

This is an example of cvflow working with detection/segmentation/classification
models.
Example of running a script:  
Usbcam:
```angular2html
python3 [script name] --usbcam /dev/video0 --font ../fonts/OpenSans-VariableFont_wdth,wght.ttf --fsize 1280,1920
```   
Videofile:
```angular2html
python3 [script name] --videofile {file_path},{file_path} --font ../fonts/OpenSans-VariableFont_wdth,wght.ttf --fsize 1280,1920
```

### Commands list

* --usbcam <path_to_device>,<path_to_device>,... - list of sources for reading the video stream from the camera   
* --videofile <path_to_file>,<path_to_file>,... - list of sources for reading a video stream from a video file 
* -c or --confidence <threshold_value> - threshold value for the model 
* -f or --font <path_to font> - the path to the font file. File format "*.ttf"   
* --tsize - <n, k> - the format of the video stream grid on the output.   
* --fsize <H, W> - the resolution of the image to output.   
* -d or --device <device_type> - The device on which the conveyor will work. Available: cpu, cuda.
* -l or --line - coordinates of the line intersecting the objects to be counted.


### Stereo distance tracker example

#### Creating ROI-bounding boxes
1. Run giu/roi_getter.py with parameters: -n to set number of objects and -f to set filepath.
2. Press 'd' to skip 10 frames
3. Press 's' to start selection
4. Press 'n' to add each bbox. When you will select n objects, conf.txt  will be written automatically.

#### Running
5. Download the [weights](https://drive.google.com/file/d/1FM7rTKbpYWQ1-QUzQoMPWSRDEn_LBx6l/view?usp=sharing) and put it to tests/test_data folder.
6. Run stereo_example.py for demonstration. You can use data from tests/test_data for example.
NB! Model works only with cuda. Please provide at least two data sources. 

### Objects counter example

#### Create configuration of counting lines
1. Run command:
```
python3 config_gui.py -v <video_file> -n <path_to_config_file>
```
2. Press save and close the window.

#### Running
```
python3 objects_counter_example.py --videofile <path_to_video> -f <path_to_fonts> -c <path_to_config_file>
```

### Plates detection example
1. Run command:
```
python3 config_gui.py -v <video_file> -n <path_to_config_file>
```
2. Press save and close the window.

#### Running
```
python3 examples/plates_example/run.py --videofile <path_to_video> -f <path_to_fonts> -c <path_to_config_file>
```


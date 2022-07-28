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
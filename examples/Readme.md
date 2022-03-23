# Examples

## detection_example
This is an example of cvflow working with detection models.  
Example of running a script:  
Usbcam:
```angular2html
python3 detection_example.py --usbcam /dev/video0 --font ../fonts/OpenSans-VariableFont_wdth,wght.ttf --tsize 1280,1920
```   
Videofile:
```angular2html
python3 detection_example.py --videofile {file_path},{file_path} --font ../fonts/OpenSans-VariableFont_wdth,wght.ttf --tsize 1280,1920
```

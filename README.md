![Tests](https://github.com/MLFreelib/cvflow/workflows/Tests/badge.svg)
# cvflow

### Stereo distance tracker example

1. Download the [weights](https://drive.google.com/file/d/1FM7rTKbpYWQ1-QUzQoMPWSRDEn_LBx6l/view?usp=sharing) and put it to tests/test_data folder.
2. Run stereo_example.py for demonstration.
3. Replace readers with your own data.

#### Creating ROI-bounding boxes
1. Run giu/roi_getter.py with parameters: -n to set number of objects and -f to set filepath.
2. Press 'd' to skip 10 frames
3. Press 's' to start selection
4. Press 'n' to add each bbox. When you will select n objects, conf.txt  will be written automatically.
NB! Model works only with cuda. Please provide at least two data sources. 
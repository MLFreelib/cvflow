# API of components

## model_base
package with a basic component

### ComponentBase
base class for all components.

#### params:
* name: str - name of component;

#### methods:
* get_name - Returns the name of component.
  * Parameters: Empty
  * Returns: 
    * name of component
  * Return type - str

* get_source_names - Returns the source names. Frames from each input source will be processed in this model.
  * Parameters: Empty.
  * Returns:
    * source names.
  * Return type - List[str]

* set_source_names - Sets the source names. Frames from each input source will be processed in this model.
  * Parameters:
    * source_names: List[str] - source names
  * Returns: None

* set_device - Sets the type of device on which this component will work.
  * Parameters:
    * device: str - device. Valid values: cuda, cpu.
  * Returns: None
  * Exceptions:
    * TypeError if an incorrect device type is passed.

* get_device - Returns the installed device type on which this component is running.
  * Parameters: Empty.
  * Returns:
    * type of device. Valid values: cuda, cpu.
  Return type:
    * str

* do - The main method used to implement the functionality of the component.
  * Parameters: 
    * data: MetaBatch - a block of information about data.
  * Returns:
    * a block of information about data.
  * Return type:
    * MetaBatch

* start - The method that is executed when compiling the pipeline.
  * Parameters: Empty.
  * Returns: None

* stop - The method starts when the pipeline stops. This is necessary to delete data and close windows.
  * Parameters: Empty.
  * Returns: None

## reader_component  
Readers components are used to receive a video stream from various sources.

### CamReader
CamReader allows you to read data from USB cameras.

#### params:
* device: (str | int) - path to USB-camera;
* name: str - name of component;

#### methods:
* All methods from ComponentBase;
* run - allows the component to start reading data.
  * Parameters: Empty
  * Returns: None

* read - returns the frame read at the time of the method call. If the frame could not be read, it returns the last successfully read frame.
  * Parameters: Empty.
  * Returns - Returns an image.
  * Return type - np.ndarray[H, W, C]

* stop - stops reading the video stream.
  * Parameters: Empty.
  * Returns: None

### VideoReader
CamReader allows you to read data from USB cameras.

#### params:
* device: (str | int) - path to video file;
* name: str - name of component;

#### methods:
* All methods from ComponentBase;
* run - allows the component to start reading data. And checks for the presence of a file at the specified path.
  * Parameters: Empty
  * Returns: None
  * Exceptions:
    * TypeError if the file could not be read.

* read - returns the frame read at the time of the method call. If the frame could not be read, it returns the last successfully read frame.
  * Parameters: Empty.
  * Returns - Returns an image.
  * Return type - np.ndarray[H, W, C]

* stop - stops reading the video stream.
  * Parameters: Empty.
  * Returns: None

## muxer_component  

### SourceMuxer
Container for components of the ReaderBase type. An important component for the pipeline.


#### params:
* name: str - name of component;
* max_batch_size: (int) - batch size for each reader;

#### methods:
* All methods from ComponentBase;
* add_source - Add reader to muxer.
  * Parameters: 
    * source: ReaderBase - the reader to be added to muxer
  * Returns: None
  * Exceptions:
    * TypeError if source type is not ReaderBase.

* read - returns the frame read at the time of the method call. If the frame could not be read, it returns the last successfully read frame.
  * Parameters: Empty.
  * Returns - Returns an image.
  * Return type - np.ndarray[H, W, C]

* stop - stops reading the video stream.
  * Parameters: Empty.
  * Returns: None

## model_component 

### ModelBase
Component of basic model. This class is necessary for implementing models using inheritance.
- boxes - [N, 4]
- labels - [N]
- scores - [N]

#### params:
* name: str - name of component;
* model: torch.nn.Module - the model that will process the images inside the component.

#### methods:
* All methods from ComponentBase;
* set_confidence - Setting the confidence threshold.
  * Parameters: 
    * conf: float [0-1] - value of threshold
  * Returns: None

* start - Specifies the device on which the model will be executed.
  * Parameters: Empty.
  * Returns - None.

* stop - deletes data from memory.
  * Parameters: Empty.
  * Returns: None

* set_transforms - Method of setting transformations for frames that are passed to the model.
  * Parameters: 
    * tensor_transforms: list - list of transformations from torchvision.transforms
  * Returns: None

* add_source - Names of input sources from which data will be processed by the component.
  * Parameters:
    * name: str - name of source
  * Returns: None

* set_labels - Sets labels for model.
  * Parameters:
    * labels: List[str] - list of labels.
  * Returns: None

* get_labels - Returns the label names.
  * Parameters: Empty
  * Returns:
    * self.__label_names - List[str] - label names.

### ModelDetection
Component for detection models.  
The model must have a forward method that returns a dictionary with the keys:
- boxes - [N, 4]
- labels - [N]
- scores - [N]

#### params:
* name: str - name of component;
* model: torch.nn.Module - detection model.

#### methods:
* All methods from ModelBase;
* do - Transmits data to the detection model. And adds the predicted bounding boxes with labels to the MetaFrame 
in the MetaBatch. Bounding boxes in BBoxMeta and labels in MetaLabel, which are contained in BBoxMeta.
  * Parameters: 
    * data: MetaBatch - data with information about the frames.
  * Returns: 
    * data: MetaBatch - data with information about the frames + MetaBBox inside.

### ModelClassification
Component for classification models.

#### params:
* name: str - name of component;
* model: torch.nn.Module - classification model, which returns vector of shape [N, K], where N - batch size, K - number 
of labels and values in the range from 0 to 1.

#### methods:
* All methods from ModelBase;
* do - Transmits data to the classification model. And adds the predicted labels to the MetaFrame
in the MetaBatch. Labels in MetaLabel, which are contained in MetaFrame.
  * Parameters: 
    * data: MetaBatch - data with information about the frames.
  * Returns: 
    * data: MetaBatch - data with information about the frames + MetaLabel inside.

### ModelSegmentation
Component for segmentation models. 

#### params:
* name: str - name of component;
* model: torch.nn.Module - segmentation model, which returns dictionary with key "out" which contains tensor of shape 
[N, K, H, W], where N - batch size, K - number of labels, H - mask height, W - mask width and values in the range from 0 to 1.

#### methods:
* All methods from ModelBase;
* do - Transmits data to the segmentation model. And adds the predicted masks with labels to the MetaFrame
            in the MetaBatch. Masks in MetaMask and labels in MetaLabel, which are contained in MetaMask.
  * Parameters: 
    * data: MetaBatch - data with information about the frames.
  * Returns: 
    * data: MetaBatch - data with information about the frames + MetaMask inside.

### ModelSegmentation
Component for segmentation models. 

#### params:
* name: str - name of component;
* model: torch.nn.Module - stereo model, which returns dictionary with key "out" which contains tensor of shape
[N, H, W], where N - batch size, H - mask height, W - mask width and values in the range from 0 to 1.

#### methods:
* All methods from ModelBase;
* do - Transmits data to the depth model. And adds the depth map to the MetaFrame in the MetaBatch. The depth map 
is stored in MetaDepth.
  * Parameters: 
    * data: MetaBatch - data with information about the frames.
  * Returns: 
    * data: MetaBatch - data with information about the frames + MetaDepth inside.

## tracker_component

### TrackerBase
Component for segmentation models. 

#### params:
* name: str - name of component;
* model: torch.nn.Module - stereo model, which returns dictionary with key "out" which contains tensor of shape
[N, H, W], where N - batch size, H - mask height, W - mask width and values in the range from 0 to 1.

#### methods:
* All methods from ModelBase;
* do - Transmits data to the depth model. And adds the depth map to the MetaFrame in the MetaBatch. The depth map 
is stored in MetaDepth.
  * Parameters: 
    * data: MetaBatch - data with information about the frames.
  * Returns: 
    * data: MetaBatch - data with information about the frames + MetaDepth inside.


## handler_component

### Counter
Draws a line and counts objects by ID that intersect this line.

#### params:
* name: str - name of component;
* line: List[int] - the line along which the objects will be counted. Format: [x_min, y_min, x_max, y_max]

#### methods:
* All methods from ModelBase;
* do - Counts objects.
  * Parameters: 
    * data: MetaBatch - data with information about the frames.
  * Returns: 
    * data: MetaBatch - data with information about the frames + CustomMeta "counter".


### Filter
Filters labels in metadata, leaving only those that are passed to the component.

#### params:
* name: str - name of component;
* labels: List[str] - the names of the labels that need to be left.

#### methods:
* All methods from ModelBase;
* do - Filters labels.
  * Parameters: 
    * data: MetaBatch - metadata about batch.
  * Returns: 
    * data: MetaBatch - data with information about frames without metadata that is not included in the labels passed 
to the method.

### DistanceCalculator
Calcucates distance in mm using depth.

#### params:
* name: str - name of component.

#### methods:
* All methods from ModelBase;
* do - Based on the depth map, calculates the distance between objects in millimeters.  Distance information is also
applied to the image.
  * Parameters: 
    * data: MetaBatch - metadata about batch.
  * Returns: 
    * data: MetaBatch - data with information about frames + modified frames.

## painter_component

### Tiler
Component which combines frames from different sources into one frame in the form of a grid.

#### params:
* name: str - name of component;
* tiler_size: tuple[int, int] - number of rows and columns. Example: (3, 2) for 6 frames. If there are not enough
frames, then the remaining space is filled black.

#### methods:
* All methods from ModelBase;
* do - Combines frames from different sources into one frame in the form of a grid.
  * Parameters: 
    * data: MetaBatch - data with information about the frames.
  * Returns: 
    * data: MetaBatch - data with information about the frames + new source 'tiler'


### BBoxPainter
A component for drawing bounding boxes on frames.

#### params:
* name: str - name of component;
* font_path: str - path to font;
* font_size: int - font size;
* font_width: int - font width.

#### methods:
* All methods from ModelBase;
* do - Draws bounding boxes with labels on frames.
  * Parameters: 
    * data: MetaBatch - metadata about batch.
  * Returns: 
    * data: MetaBatch - data with information about frames + modified frames with bounding boxes.
* set_font_size - sets the font size of the labels.
  * Parameters: 
    * font_size: int - font size.
  * Returns: None
* set_font - sets the font of the labels.
  * Parameters: 
    * font_path: str - path to font. ttf format.
  * Returns: None
* set_font_width - sets the font width of the labels.
  * Parameters: 
    * font_width: int - font width.
  * Returns: None

### LabelPainter
Writes a label to an image.

#### params:
* name: str - name of component

#### methods:
* All methods from ModelBase;
* do - Writes labels on the frames.
  * Parameters: 
    * data: MetaBatch - metadata about batch.
  * Returns: 
    * data: MetaBatch - data with information about frames + modified frames with label.
* set_org - sets the coordinates of the text on the image.
  * Parameters: 
    * org: tuple[int, int] - position of label.
  * Returns: None
* set_colors - sets the colors for the labels.
  * Parameters: 
    * colors: dict[str, tuple[int, int, int]] - colors of the labels. Example: {'lion': (0, 255, 255)
  * Returns: None
* set_thickness - sets the thickness of the text for labels.
  * Parameters: 
    * thickness: int - font thickness: int.
  * Returns: None
* set_lineType - sets the line type of the text for labels.
  * Parameters: 
    * lineType: int - line type of text. Values: [0-7, 16]
  * Returns: None
* set_font_scale - sets the scale of the text for labels.
  * Parameters: 
    * font_scale: int - font scale.
  * Returns: None

### MaskPainter
A component for drawing masks on frames.

#### params:
* name: str - name of component

#### methods:
* All methods from ModelBase;
* do - Draws masks on frames.
  * Parameters: 
    * data: MetaBatch - metadata about batch.
  * Returns: 
    * data: MetaBatch - data with information about frames + modified frames with masks.
* set_alpha - sets the transparency of masks.
  * Parameters: 
    * alpha: float - transparency parameter.
  * Returns: None

### DepthPainter
A component for drawing depth maps on frames.

#### params:
* name: str - name of component

#### methods:
* All methods from ModelBase;
* do - Draws depth maps on frames.
  * Parameters: 
    * data: MetaBatch - metadata about batch.
  * Returns: 
    * data: MetaBatch - data with information about frames + modified frames with depth maps.
* set_alpha - sets the transparency of depth maps.
  * Parameters: 
    * alpha: float - transparency parameter.
  * Returns: None

## outer_component

### DisplayComponent
A component for displaying the pipeline result.

#### params:
* name: str - name of component;
* escape_btn: str -the key to close the window. Default key: 'q'.

#### methods:
* All methods from ModelBase;
* do - Displays frames from selected sources in the window.
  * Parameters: 
    * data: MetaBatch - data with information about the frames.
  * Returns: 
    * data: MetaBatch - data with information about the frames.

### FileWriterComponent
A component for writing the pipeline result to file.

#### params:
* name: str - name of component;
* file_path: str - the path to the file where the result will be saved;
* framerate: int - the framerate with which the result will be saved;
* fourcc: str - fourcc code of the codec. Default value: 'XVID'.

#### methods:
* All methods from ModelBase;
* do - Puts the frames to the queue.
  * Parameters: 
    * data: MetaBatch - data with information about the frames.
  * Returns: 
    * data: MetaBatch - data with information about the frames.

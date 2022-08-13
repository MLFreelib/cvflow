# API of Meta

The structure of the data with which the components of the framework work.  

## MetaBatch
A container for storing a batch of frames.

### params:
* name: str - name of structure;

### methods:
* add_signal - Adds name to signals.
  * Parameters:
    * name: str - name of signal.
  * Exceptions:
    * TypeError if type of name is not str.

* set_signal - Sets the value for signals by name.
  * Parameters: 
    * name: str - name of signal;
    * value: Any - value of signal.
  * Exceptions:
    * TypeError if name of signal is not str;
    * ValueError if the signal name is not found.

* get_signal - Returns the value by name.
  * Parameters:
    * name: str - name of signal
  * Returns: Any - value of signal.

* add_meta_frame - Adds a frame with information about this frame to the batch.
  * Parameters:
    * frame: MetaFrame - adds MetaFrame to MetaBatch.
  * Exceptions: 
    * TypeError if type of frame is not MetaFrame.

* add_frames - Adds a frames to the batch.
  * Parameters:
    * name: str - name of source.
  * Exceptions:
    * TypeError if type of frames is not tensor.

* get_frames_by_src_name - Returns frames received from a specific source.
  * Parameters:
    * src_name: str - name of source.
  * Returns: Union[torch.tensor, None] - frames by source name.

* get_frames_all - Returns all frames.
  * Returns: Dict[str, torch.tensor] - all frames, where key is name of source and values is frames.

* get_meta_frames_by_src_name - Returns information about frames by the source name.
  * Parameters:
    * src_name: str - name of source.
  * Returns: Union[List[MetaFrame], None] - MetaFrame's by source name.

* get_meta_frames_all - Returns information about all frames in a batch.
  * Returns: Dict[str, MetaFrame] - all MetaFrame's, where key is name of source and values is information about frames.

* set_source_names - Sets the source names.
  * Parameters:
    * source_names: List[str] - source names.
  * Exceptions:
    * TypeError if source name is not list.

* get_source_names - Returns the names of all the sources from which the frames were received.
  * Parameters:
    * src_name: str - name of source.
  * Returns: List[str] - source names.

## MetaFrame
Container for storing a frame information.

### params:
* source_name: str - the name of the source from which the frame was received;
* frame: torch.tensor - image. Expected shape [3, H, W].

### methods:
* get_src_name - Returns the name of the source from which the frame was received.
  * Returns: str - source name.
  * Exceptions:
    * TypeError if type of name is not str.

* set_frame - Sets the value for signals by name.
  * Parameters: 
    * name: frame: torch.tensor - image. Expected shape [3, H, W] or [1, H, W].
  * Exceptions:
    * TypeError if type of frame is not tensor;
    * ValueError if the number of channels is not equal to 3.
    * ValueError if the format of image is not [3, H, W] or [1, H, W].

* add_meta - Returns the value by name.
  * Parameters:
    * meta_name: str - name of custom data.
    * value: Any - custom data.

* get_frame - Returns a frame.
  * Returns: torch.Tensor - image.

* get_meta_info - Returns custom data by name.
  * Parameters:
    * name: str - name of custom data.
  * Returns: Any - custom Meta data.

* set_label_info - Sets information about the label in the frame.
  * Parameters:
    * labels_info: MetaLabel - meta data for labels.
  * Exceptions:
    * TypeError if type of label_info is not MetaLabel.

* get_labels_info - Returns a MetaLabel that contains information about the labels in the frame.
  * Returns: MetaLabel - meta data about labels.

* set_mask_info - Sets information about predicted masks for this frame.
  * Parameters:
    * mask_info: MetaMask - meta data for masks.
  * Exceptions:
    * TypeError if type of label_info is not MetaMask.

* get_mask_info - Returns the predicted masks for this frame.
  * Returns: MetaMask - meta data about masks.

* set_bbox_info - Sets information about the label in the frame.
  * Parameters:
    * bbox_info: MetaBBox - meta data for bounding boxes.
  * Exceptions:
    * TypeError if type of label_info is not MetaBBox.

* get_bbox_info - Returns the predicted bounding boxes for this frame.
  * Returns: MetaMask - meta data about bounding boxes.

* set_depth_info - Sets information about predicted depth for this frame.
  * Parameters:
    * depth_info: MetaDepth - meta data for depth maps.
  * Exceptions:
    * TypeError if type of label_info is not MetaDepth.

* get_depth_info - Returns the predicted depth for this frame.
  * Returns: MetaDepth - meta data about depth maps.


## MetaLabel
Container for storing information about labels and id from tracking.

### params:
* labels: List[str] - list of label names;
* confidence: List[float] - confidence in the label for each label from labels.

### methods:
* get_confidence - Returns tensor of confidences for each label.
  * Returns: Tensor - tensor of confidence for each label.

* set_object_id - Sets the ids for each label.
  * Parameters: 
    * object_ids: List[int] - list of ids for each object.
  * Exceptions:
    * ValueError if the number of objects ids is not equal to the number of labels.

* get_labels - Returns a list of predicted labels.
  * Returns: List[str] - list of labels.

* get_object_ids - Returns a list of id for each label.
  * Returns: List[int] - object ids for each label.

## MetaBBox
Container for storing information about bounding boxes.

### params:
* points: torch.tensor - bounding boxes with shape: [N, 4]. Bounding box format: [x_min, y_min, x_max, y_max];
* label_info: MetaLabel - information about each bounding box.

### methods:
* get_bbox - Returns the bounding boxes.
  * Returns: Tensor - tensor of bounding boxes. Bounding box format: [x_min, y_min, x_max, y_max].

* set_bboxes - Sets the bounding boxes with shape: [N, 4] and format: [x_min, y_min, x_max, y_max].
  * Parameters:
    * points: torch.tensor - tensor of bounding boxes.
  * Exceptions:
    * TypeError if the type of bounding boxes is not torch.tensor.
    * ValueError if the bbox shape is not equal to the 2.
    * ValueError if the bbox size is not equal to the 4.

* get_label_info - Returns a MetaLabel that contains information about the labels for each bounding box.
  * Returns: MetaLabel - metadata about labels for each bounding box.

* set_label_info - Sets the metadata about labels for each bounding box.
  * Parameters:
    * label_info: MetaLabel - metadata about labels for each bounding box.
  * Exceptions:
    * ValueError if number of bounding boxes is not equal to the number of labels.

## MetaMask
Container for storing information about masks

### params:
* mask: torch.tensor - batch of masks;
* label_info: MetaLabel - information about each mask.

### methods:
* get_mask - Returns the batch of masks.
  * Returns: torch.tensor - batch of masks.

* set_mask - Sets the masks with shape [N, K, H, W]. N - number of frames, K - number of labels, H - height, W - width.
  * Parameters:
    * mask: torch.tensor - tensor of masks.
  * Exceptions:
    * ValueError if the masks shape is not equal to [N, K, H, W].

* get_label_info - Returns a MetaLabel that contains information about the labels for each mask.
  * Returns: MetaLabel - metadata about labels for each mask.

* set_label_info - Sets the metadata about labels for each mask.
  * Parameters:
    * label_info: MetaLabel - metadata about labels for each mask.
  * Exceptions:
    * ValueError if number of masks is not equal to the number of labels.

## MetaDepth
Container for storing information about depth

### params:
* depth: torch.tensor - batch of depth masks.

### methods:
* get_depth - Returns the batch of depth.
  * Returns: torch.tensor - batch of depth.

* set_depth - Sets the depth with shape [N, H, W]. N - number of frames, H - height, W - width.
  * Parameters:
    * depth: torch.tensor - tensor of depth.
  * Exceptions:
    * ValueError if the masks shape is not equal to [N, H, W].

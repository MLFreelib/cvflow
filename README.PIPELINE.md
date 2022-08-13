# API of Pipeline

The structure of the data with which the components of the framework work.  

## Pipeline
A container for building and controlling the pipeline. Allows you to manage components, start and stop them.

### params:
* name: str - name of structure;

### methods:
* add - Adds a component to the pipeline.
  * Parameters:
    * component: ComponentBase - the component that will be in the pipeline.
  * Exceptions:
    * InvalidComponentException if first component is not MuxerBase.

* set_device -  Sets the device type. Available types: cpu and cuda.
  * Parameters: 
    * device: str - the device type. Available types: cpu and cuda.
  * Exceptions:
    * TypeError if device type is not cuda or cpu;

* add_all - Adds a list of components to the pipeline. The data in the pipeline will move through the components 
in the order in which they are in the list.
  * Parameters:
    * components: List[ComponentBase] - list of components that will be in the pipeline.
  * InvalidComponentException if first component is not MuxerBase.
  * TypeError if component is not ComponentBase.

* run - Starts a loop that moves data between components in the pipeline.

* compile - Configures and verifies components.

* close - Closes each component.

* add_signals - Adds new signal to the pipeline.
  * Parameters:
    * signals: List[str] - list of new signals

### 1. Create Data set
At the beginning we try to capture images preprocessed by removing background, 
but experiment showed that quality of image is under satisfaction.  
So our collected data just captured by laptop camera with a white wall as background, 
and we also turned those images into binary mode.

### 2. After roughly trained a MobileNetv2
We just trained about 30 epochs and the accuracy on training set seems good, which reached 99%.
However, after testing we found that the model failed to classify the class `nothing`, which probably caused by poor training data with less variance.
So next step before fune-tuning or implement other NN, we have to refine our training data set.

### 3. About our control pipeline
Obviously, we have to use multi-threading for gesture recognition and controls.

We have following functions to implement:
- handle video input stream 
- dynamic visualization of model prediction
- use the predictions to generate instructions for car control



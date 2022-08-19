# UV coordinates prediction for human figures from pictorial maps

This is a simple fully convolutional network for predicting 
UV coordinates of figures from depth images and body part masks.

## Usage

### Training

* Set TRAINING_DATA_FOLDER in config.py to our training data.
* Run generator.py (it will call train_and_eval.py).
* Optionally, adapt some parameters in config.py.

### Inference

* Run inference.py to run the network with an existing model.

### Miscellaneous

* Run visualize.py to show greyscale depth images, colored UV images and body parts masks.

## Configurations

* Change INCLUDE_BODY_PARTS in config.py to train without and with body part masks.

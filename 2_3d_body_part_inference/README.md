# 3D body part inference for human figures from pictorial maps

This is a adapted reimplementation of https://github.com/laughtervv/DISN for creating 
3D body part SDFs based on their 2D silhouettes.

## Usage

### Training

* Set the input_folder in generator.py to our training data.
* Run generator.py (it will call train_and_eval.py).
* Optionally adapt some parameters in config.py.

### Inference

* Run inference.py to run the network with an existing model.

### Miscellaneous

* Run statistics.py to average the results of different runs.
* Run visualize.py to show 3D occupancy fields and their 2D projections.

## Configurations

* DISN one-stream or two-stream without or with pose points
* See details in models.py > model_parts.py > utils.py

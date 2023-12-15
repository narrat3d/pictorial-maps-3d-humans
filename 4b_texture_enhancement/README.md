# Texture resynthesis of pictorial human heads

This is a combination of two autoencoders for denoising and recovering facial details from RGB and UV images of pictorial human heads.

## Usage 

### Training

* Set train_folder and test_folder to our training and test data.

### Inference

* Set is_training = False to run the network with an existing model.

### Configurations

* Different augmentation methods like blurring, rotating, oilpainting, shifting colors, adding noises can be applied to the input images.

## Miscellaneous

* The folder `cartoonize` contains code by Wang & Yu 2020, which was used to create the target data for training.
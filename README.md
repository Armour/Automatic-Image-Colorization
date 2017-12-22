# Residual Encoder Network for Colorization

## Overview

This is a Tensorflow implementation of the Residual Encoder Network based on [Automatic Colorization](http://tinyclouds.org/colorize/) and the pre-trained VGG16 model from [https://github.com/machrisaa/tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg)

## Structure

* `config.py`: config variables like learning rate, batch size and so on
* `image_helper.py`: all functions related to image manipulation
* `read_input.py`: input related functions
* `residual_encoder.py`: the residual encoder model
* `batchnorm.py`: batch normalization based on [this](http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow)
* `common.py`: the common part for training and testing, mainly the work flow for this model
* `train.py`: train the residual encoder model using Tensorflow built-in GradientDescentOptimizer
* `test.py`: test your own images and save the output images

## Tensorflow graph

![](images/residuall_encoder.png)

## How to use

* First please download pre-trained VGG16 model [vgg16.npy](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) to vgg folder

* Use pre-trained residual encoder model
    * Model can be downloaded [here](https://github.com/Armour/Automatic-Image-Colorization/releases/tag/1.0)
    * Unzip all files to `model_path` (you can change this path in `config.py`)
    * **[UPDATE: Dec 22 2017]** Note that this model was trained under tensorflow==0.12.1, it's not really working for tensorflow 1.0+, I'll try to train a new model under tensorflow 1.4.0 and upload it soon

* Train your own model
    1. Change the `learning rate`, `batch size` and `training_iters` accordingly
    2. Change `train_dir` to your directory that contains all your training jpg images
    3. Run `python train.py`

* Test
    1. Change `test_dir` to your directory that contains all your testing jpg images
    2. Run `python test.py`

## References

* [Automatic Colorization](http://tinyclouds.org/colorize/)
* [pavelgonchar/colornet](https://github.com/pavelgonchar/colornet)
* [pretrained VGG16 npy file](https://github.com/machrisaa/tensorflow-vgg)

## License

GNU GPL 3.0 for personal or research use. COMMERCIAL USE PROHIBITED.

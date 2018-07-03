# Automatic Image Colorization

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Template from jarvis](https://img.shields.io/badge/Hi-Jarvis-ff69b4.svg)](https://github.com/Armour/Jarvis)

## Overview

This is a Tensorflow implementation of the Residual Encoder Network based on [Automatic Colorization](http://tinyclouds.org/colorize/) and the pre-trained VGG16 model from [https://github.com/machrisaa/tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg)

## Structure

* `config.py`: config variables like batch size, training_iters and so on
* `image_helper.py`: all functions related to image manipulation
* `read_input.py`: all functions related to input
* `residual_encoder.py`: the residual encoder model
* `common.py`: the common part for training and testing, which is mainly the workflow for this model
* `train.py`: train the residual encoder model using Tensorflow built-in AdamOptimizer
* `test.py`: test your own images and save the output images

## Tensorflow graph

![residual_encoder](images/residual_encoder.png)

## How to use

* First please download pre-trained VGG16 model [vgg16.npy](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) to vgg folder

* Option 1: Use pre-trained residual encoder model
  * Model can be downloaded [here](https://github.com/Armour/Automatic-Image-Colorization/releases/tag/2.0)
  * Unzip all files to `summary_path` (you can change this path in `config.py`)

* Option 2: Train your own model!
  1. Change the `batch_size` and `training_iters` if you want.
  2. Change `training_dir` to your directory that contains all your training jpg images
  3. Run `python train.py`

* Test
  1. Change `testing_dir` to your directory that contains all your testing jpg images
  2. Run `python test.py`

## Examples

* ![1](images/1.png)
* ![2](images/2.png)
* ![3](images/3.png)
* ![4](images/4.png)
* ![5](images/5.png)
* ![6](images/6.png)
* ![7](images/7.png)
* ![8](images/8.png)
* ![9](images/9.png)
* ![10](images/10.png)
* ![11](images/11.png)
* ![12](images/12.png)

* More example output images can be found in [sample_output_images](https://github.com/Armour/Automatic-Image-Colorization/blob/master/sample_output_images) folder.

## References

* [Automatic Colorization](http://tinyclouds.org/colorize/)
* [pavelgonchar/colornet](https://github.com/pavelgonchar/colornet)
* [raghavgupta0296/ColourNet](https://github.com/raghavgupta0296/ColourNet)
* [pretrained VGG16 npy file](https://github.com/machrisaa/tensorflow-vgg)

## Contributing

See [CONTRIBUTING.md](https://github.com/Armour/Automatic-Image-Colorization/blob/master/.github/CONTRIBUTING.md)

## License

[GNU GPL 3.0](https://github.com/Armour/Automatic-Image-Colorization/blob/master/LICENSE) for personal or research use. COMMERCIAL USE PROHIBITED.

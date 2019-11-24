#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Config file that contains all config varibles."""

__author__ = 'Chong Guo <armourcy@email.com>'
__copyright__ = 'Copyright 2018, Chong Guo'
__license__ = 'GPL'

import numpy as np
import tensorflow as tf


# Debug flag, if true, will check model shape using assert in each step and skip gray image check part (to save time)
debug = False

# Image size for training
image_size = 224

# Image resize method
image_resize_method = tf.image.ResizeMethod.BILINEAR

# Parameters for neural network
training_iters = 3000000  # The training iterations number
batch_size = 6 # Batch size for training data
display_step = 50  # Step interval for displaying loss and saving summary during training phase
testing_step = 1000  # Step interval for testing and saving image during training phase
saving_step = 10000  # Step interval for saving model during training phase
shuffle_buffer_size = 2000

# UV channel normalization parameters
u_norm_para = 0.435912
v_norm_para = 0.614777

# Directory for training and testing dataset
training_dir = "train2014"
testing_dir = "vgg/test_data"

# Model, result and generated images stored path
summary_path = "summary"
training_summary = summary_path + "/train"
testing_summary = summary_path + "/test"

# Weights for each layer (trainable)
weights = {
    'b_conv4': tf.Variable(tf.truncated_normal([1, 1, 512, 256], stddev=0.01), trainable=True),
    'b_conv3': tf.Variable(tf.truncated_normal([3, 3, 256, 128], stddev=0.01), trainable=True),
    'b_conv2': tf.Variable(tf.truncated_normal([3, 3, 128, 64], stddev=0.01), trainable=True),
    'b_conv1': tf.Variable(tf.truncated_normal([3, 3, 64, 3], stddev=0.01), trainable=True),
    'b_conv0': tf.Variable(tf.truncated_normal([3, 3, 3, 3], stddev=0.01), trainable=True),
    'output_conv': tf.Variable(tf.truncated_normal([3, 3, 3, 2], stddev=0.01), trainable=True),
}

# Gaussian blur kernel (not trainable)
gaussin_blur_3x3 = np.divide([
    [1., 2., 1.],
    [2., 4., 2.],
    [1., 2., 1.],
], 16.) # (3, 3)
gaussin_blur_3x3 = np.stack((gaussin_blur_3x3, gaussin_blur_3x3), axis=-1) # (3, 3, 2)
gaussin_blur_3x3 = np.stack((gaussin_blur_3x3, gaussin_blur_3x3), axis=-1) # (3, 3, 2, 2)

gaussin_blur_5x5 = np.divide([
    [1.,  4.,  7.,  4., 1.],
    [4., 16., 26., 16., 4.],
    [7., 26., 41., 26., 7.],
    [4., 16., 26., 16., 4.],
    [1.,  4.,  7.,  4., 1.],
], 273.) # (5, 5)
gaussin_blur_5x5 = np.stack((gaussin_blur_5x5, gaussin_blur_5x5), axis=-1) # (5, 5, 2)
gaussin_blur_5x5 = np.stack((gaussin_blur_5x5, gaussin_blur_5x5), axis=-1) # (5, 5, 2, 2)

tf_blur_3x3 = tf.Variable(tf.convert_to_tensor(gaussin_blur_3x3, dtype=tf.float32), trainable=False)
tf_blur_5x5 = tf.Variable(tf.convert_to_tensor(gaussin_blur_5x5, dtype=tf.float32), trainable=False)

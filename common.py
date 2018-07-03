#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The utility functions for training and testing."""

__author__ = 'Chong Guo'
__copyright__ = 'Copyright 2018, Chong Guo'
__license__ = 'GPL'
__email__ = 'armourcy@email.com'

import os

import tensorflow as tf
from vgg import vgg16

from config import batch_size, testing_dir, training_dir
from image_helper import rgb_to_yuv, yuv_to_rgb
from read_input import init_file_path, get_dataset_iterator
from residual_encoder import ResidualEncoder


def create_folder(folder_path):
    """
    Create folder if not exist.
    :param folder_path:
    :return: None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def init_model(train=True):
    """
    Init model for both training and testing.
    :param train: indicate if current is in training
    :return: all stuffs that need for this model
    """
    # Create training summary folder if not exist
    create_folder("summary/train/images")

    # Create testing summary folder if not exist
    create_folder("summary/test/images")

    # Use gpu if exist
    with tf.device('/device:GPU:0'):
        # Init image data file path
        print("⏳ Init input file path...")
        if train:
            file_paths = init_file_path(training_dir)
        else:
            file_paths = init_file_path(testing_dir)

        # Init training flag and global step
        print("⏳ Init placeholder and variables...")
        is_training = tf.placeholder(tf.bool, name="is_training")
        global_step = tf.train.get_or_create_global_step()

        # Load vgg16 model
        print("🤖 Load vgg16 model...")
        vgg = vgg16.Vgg16()

        # Build residual encoder model
        print("🤖 Build residual encoder model...")
        residual_encoder = ResidualEncoder()

        # Get dataset iterator
        iterator = get_dataset_iterator(file_paths, batch_size, shuffle=True)

        # Get color image
        color_image_rgb = iterator.get_next(name="color_image_rgb")
        color_image_yuv = rgb_to_yuv(color_image_rgb, "color_image_yuv")

        # Get gray image
        gray_image_one_channel = tf.image.rgb_to_grayscale(color_image_rgb, name="gray_image_one_channel")
        gray_image_three_channels = tf.image.grayscale_to_rgb(gray_image_one_channel, name="gray_image_three_channels")
        gray_image_yuv = rgb_to_yuv(gray_image_three_channels, "gray_image_yuv")

        # Build vgg model
        with tf.name_scope("vgg16"):
            vgg.build(gray_image_three_channels)

        # Predict model
        predict = residual_encoder.build(input_data=gray_image_three_channels, vgg=vgg, is_training=is_training)
        predict_yuv = tf.concat(axis=3, values=[tf.slice(gray_image_yuv, [0, 0, 0, 0], [-1, -1, -1, 1], name="gray_image_y"), predict], name="predict_yuv")
        predict_rgb = yuv_to_rgb(predict_yuv, "predict_rgb")

        # Get loss
        loss = residual_encoder.get_loss(predict_val=predict, real_val=tf.slice(color_image_yuv, [0, 0, 0, 1], [-1, -1, -1, 2], name="color_image_uv"))

        # Prepare optimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name='optimizer')

        # Init tensorflow summaries
        print("⏳ Init tensorflow summaries...")
        tf.summary.histogram("loss", loss)
        tf.summary.image("gray_image", gray_image_three_channels, max_outputs=1)
        tf.summary.image("predict_image", predict_rgb, max_outputs=1)
        tf.summary.image("color_image", color_image_rgb, max_outputs=1)

    return is_training, global_step, optimizer, loss, predict_rgb, color_image_rgb, gray_image_three_channels, file_paths

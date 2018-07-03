#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Helper functions for read input."""

__author__ = 'Chong Guo'
__copyright__ = 'Copyright 2018, Chong Guo'
__license__ = 'GPL'
__email__ = 'armourcy@email.com'

import os
import cv2
import imghdr
import tensorflow as tf

from config import debug, shuffle_buffer_size, image_size, image_resize_method


def init_file_path(directory):
    """
    Get the image file path array.
    :param directory: the directory that store images
    :return: an array of image file path
    """
    paths = []

    if not debug:
        print("Throwing all gray space images now... (this will take a long time if the training dataset is huge)")

    for file_name in os.listdir(directory):
        # Skip files that is not jpg
        file_path = '%s/%s' % (directory, file_name)
        if imghdr.what(file_path) is not 'jpeg':
            continue
        if not debug:
            # Delete all gray space images
            is_gray_space = True
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if len(img.shape) == 3 and img.shape[2] == 3:
                for w in range(img.shape[0]):
                    for h in range(img.shape[1]):
                        r, g, b = img[w][h]
                        if r != g != b:
                            is_gray_space = False
                        if not is_gray_space:
                            break
                    if not is_gray_space:
                        break
            if is_gray_space:
                try:
                    os.remove(file_path)
                except OSError as e:
                    print ("Error: %s - %s." % (e.filename, e.strerror))
                continue
        paths.append(file_path)

    return paths


def read_image(filename):
    """
    Read and store image with RGB color space.
    :param filename_queue: the filename queue for image files
    :return: image with RGB color space
    """
    # Read image file
    content = tf.read_file(filename)
    # Decode the image with RGB color space
    rgb_image = tf.image.decode_jpeg(content, channels=3, name="color_image_original")
    # Resize image to the right image_size
    rgb_image = tf.image.resize_images(rgb_image, [image_size, image_size], method=image_resize_method)
    # Map all pixel element value into [0, 1]
    return tf.clip_by_value(tf.div(tf.cast(rgb_image, tf.float32), 255), 0.0, 1.0, name="color_image_in_0_1")


def get_dataset_iterator(filenames, batch_size, num_epochs=None, shuffle=False):
    """
    Dataset input which shuffle the input data and returns an iterator to get batches of images
    :param filenames: filenames
    :param batch_size: batch size
    :param num_epochs: number of epochs for producing each string before generating an OutOfRange error
    :param shuffle: if true, the strings are randomly shuffled within each epoch
    :return: the batch image data iterator
    """
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(read_image)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.repeat(count=num_epochs)
    iterator = dataset.make_one_shot_iterator()
    return iterator

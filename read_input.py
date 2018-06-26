#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Helper functions for read input."""

import os
import cv2
import imghdr
import tensorflow as tf

from config import debug, dequeue_buffer_size, image_size, input_resize_method


def init_file_path(directory):
    """
    Get the image file path array
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


def read_image(filename_queue):
    """
    Read and store image with RGB color space
    :param filename_queue: the filename queue for image files
    :return: image with RGB color space
    """
    # Read the image with RGB color space
    reader = tf.WholeFileReader()
    _, content = reader.read(filename_queue)
    rgb_image = tf.image.decode_jpeg(content, channels=3, name="color_image_original")
    # Resize image to the right image_size
    rgb_image = tf.image.resize_images(rgb_image, [image_size, image_size], method=input_resize_method)
    # Map all pixel element value into [0, 1]
    return tf.clip_by_value(tf.div(tf.cast(rgb_image, tf.float32), 255), 0.0, 1.0, name="color_image_in_0_1")


def input_pipeline(filenames, b_size, num_epochs=None, shuffle=False, test=False):
    """
    Use a queue that randomizes the order of examples and return batch of images
    :param filenames: filenames
    :param b_size: batch size
    :param num_epochs: number of epochs for producing each string before generating an OutOfRange error
    :param shuffle: if true, the strings are randomly shuffled within each epoch
    :param test: if true use batch, else use shuffle batch
    :return: a batch of yuv_images
    """
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=shuffle)
    rgb_image = read_image(filename_queue)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    min_after_dequeue = dequeue_buffer_size
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    capacity = min_after_dequeue + 3 * b_size
    if test:
        image_batch = tf.train.batch([rgb_image],
                                     batch_size=b_size,
                                     capacity=capacity)
    else:
        image_batch = tf.train.shuffle_batch([rgb_image],
                                             batch_size=b_size,
                                             capacity=capacity,
                                             min_after_dequeue=min_after_dequeue)
    return image_batch

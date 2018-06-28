#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Helper functions for image manipulation."""

import numpy as np
import tensorflow as tf


def rgb_to_yuv(rgb_image, scope):
    """
    Convert image color space from RGB to YUV.
    :param rgb_image: an image with RGB color space
    :param scope: scope for this function
    :return: an image with YUV color space
    """
    with tf.name_scope(scope):
        # Map from 0.0 ~ 1.0 (float) to 0 ~ 255 (int).
        rgb_image_256 = tf.cast(tf.multiply(rgb_image, 255), tf.int16)

        # Get r, g, b channel.
        _r = tf.slice(rgb_image_256, [0, 0, 0, 0], [-1, -1, -1, 1])
        _g = tf.slice(rgb_image_256, [0, 0, 0, 1], [-1, -1, -1, 1])
        _b = tf.slice(rgb_image_256, [0, 0, 0, 2], [-1, -1, -1, 1])

        # Calculate y, u, v channel.
        # https://docs.microsoft.com/en-us/previous-versions/windows/embedded/ms893078(v=msdn.10)
        _y = tf.add(tf.bitwise.right_shift(tf.add(tf.add_n([tf.multiply(_r, 66), tf.multiply(_g, 129), tf.multiply(_b, 25)]), 128), 8), 16)
        _u = tf.add(tf.bitwise.right_shift(tf.add(tf.add_n([tf.multiply(_r, -38), tf.multiply(_g, -74), tf.multiply(_b, 112)]), 128), 8), 128)
        _v = tf.add(tf.bitwise.right_shift(tf.add(tf.add_n([tf.multiply(_r, 112), tf.multiply(_g, -94), tf.multiply(_b, -18)]), 128), 8), 128)

        # Map y from 16 ~ 235 to 0.0 ~ 1.0, u and v from 16 ~ 240 to 0.0 ~ 1.0.
        _y = tf.clip_by_value(tf.div(tf.cast(tf.subtract(_y, 16), tf.float32), 219.), 0.0, 1.0)
        _u = tf.clip_by_value(tf.div(tf.cast(tf.subtract(_u, 16), tf.float32), 224.), 0.0, 1.0)
        _v = tf.clip_by_value(tf.div(tf.cast(tf.subtract(_v, 16), tf.float32), 224.), 0.0, 1.0)

        # Get image with YUV color space.
        return tf.concat(axis=3, values=[_y, _u, _v])

def yuv_to_rgb(yuv_image, scope):
    """
    Convert image color space from YUV to RGB.
    :param yuv_image: an image with YUV color space
    :param scope: scope for this function
    :return: an image with RGB color space
    """
    with tf.name_scope(scope):
        # Get y, u, v channel.
        _y = tf.slice(yuv_image, [0, 0, 0, 0], [-1, -1, -1, 1])
        _u = tf.slice(yuv_image, [0, 0, 0, 1], [-1, -1, -1, 1])
        _v = tf.slice(yuv_image, [0, 0, 0, 2], [-1, -1, -1, 1])

        # Map y from 0.0 ~ 1.0 to 16 ~ 235, u and v from 0.0 ~ 1.0 to 16 ~ 240.
        _y = tf.clip_by_value(tf.cast(tf.add(tf.multiply(_y, 219), 16), tf.int16), 16, 235)
        _u = tf.clip_by_value(tf.cast(tf.add(tf.multiply(_u, 224), 16), tf.int16), 16, 240)
        _v = tf.clip_by_value(tf.cast(tf.add(tf.multiply(_v, 224), 16), tf.int16), 16, 240)

        # Calculate r, g, b channel.
        # https://docs.microsoft.com/en-us/previous-versions/windows/embedded/ms893078(v=msdn.10)
        _c = tf.subtract(_y, 16)
        _d = tf.subtract(_u, 128)
        _e = tf.subtract(_v, 128)
        _r = tf.bitwise.right_shift(tf.add(tf.add_n([tf.multiply(_c, 298), tf.multiply(_e, 409)]), 128), 8)
        _g = tf.bitwise.right_shift(tf.add(tf.add_n([tf.multiply(_c, 298), tf.multiply(_d, -100), tf.multiply(_e, -208)]), 128), 8)
        _b = tf.bitwise.right_shift(tf.add(tf.add_n([tf.multiply(_c, 298), tf.multiply(_d, 516)]), 128), 8)

        # Get image with RGB color space.
        return tf.clip_by_value(tf.div(tf.cast(tf.concat(axis=3, values=[_r, _g, _b]), tf.float32), 255), 0.0, 1.0)

def concat_images(img_a, img_b):
    """
    Combines two color image side-by-side.
    :param img_a: image a on left
    :param img_b: image b on right
    :return: combined image
    """
    height_a, width_a = img_a.shape[:2]
    height_b, width_b = img_b.shape[:2]
    max_height = np.max([height_a, height_b])
    total_width = width_a + width_b
    new_img = np.zeros(shape=(max_height, total_width, 3), dtype=np.float32)
    new_img[:height_a, :width_a] = img_a
    new_img[:height_b, width_a:total_width] = img_b
    return new_img

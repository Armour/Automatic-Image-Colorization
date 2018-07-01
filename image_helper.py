#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Helper functions for image manipulation."""

__author__ = 'Chong Guo'
__copyright__ = 'Copyright 2018, Chong Guo'
__license__ = 'GPL'
__email__ = 'armourcy@email.com'

import numpy as np
import tensorflow as tf
from config import u_norm_para, v_norm_para


def rgb_to_yuv(rgb_image, scope):
    """
    Convert image color space from RGB to YUV.
    :param rgb_image: an image with RGB color space
    :param scope: scope for this function
    :return: an image with YUV color space
    """
    with tf.name_scope(scope):
        # Get r, g, b channel.
        _r = tf.slice(rgb_image, [0, 0, 0, 0], [-1, -1, -1, 1])
        _g = tf.slice(rgb_image, [0, 0, 0, 1], [-1, -1, -1, 1])
        _b = tf.slice(rgb_image, [0, 0, 0, 2], [-1, -1, -1, 1])

        # Calculate y, u, v channel.
        # https://www.pcmag.com/encyclopedia/term/55166/yuv-rgb-conversion-formulas
        _y = 0.299 * _r + 0.587 * _g + 0.114 * _b
        _u = 0.492 * (_b - _y)
        _v = 0.877 * (_r - _y)

        # Normalize u, v channel.
        _u = _u / (u_norm_para * 2) + 0.5
        _v = _v / (v_norm_para * 2) + 0.5

        # Get image with YUV color space.
        return tf.clip_by_value(tf.concat(axis=3, values=[_y, _u, _v]), 0.0, 1.0)

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

        # Denormalize u, v channel.
        _u = (_u - 0.5) * u_norm_para * 2
        _v = (_v - 0.5) * v_norm_para * 2

        # Calculate r, g, b channel.
        # https://www.pcmag.com/encyclopedia/term/55166/yuv-rgb-conversion-formulas
        _r = _y + 1.14 * _v
        _g = _y - 0.395 * _u - 0.581 * _v
        _b = _y + 2.033 * _u

        # Get image with RGB color space.
        return tf.clip_by_value(tf.concat(axis=3, values=[_r, _g, _b]), 0.0, 1.0)

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

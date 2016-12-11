"""
Helper functions for image manipulation
"""

import numpy as np

from config import *


def rgb_to_yuv(rgb_image, scope):
    """
    Convert image color space from RGB to YUV
    :param rgb_image: an image with RGB color space
    :param scope: scope for this function
    :return: an image with YUV color space
    """
    with tf.name_scope(scope):
        # Get r, g, b channel
        _r = tf.slice(rgb_image, [0, 0, 0, 0], [-1, -1, -1, 1])
        _g = tf.slice(rgb_image, [0, 0, 0, 1], [-1, -1, -1, 1])
        _b = tf.slice(rgb_image, [0, 0, 0, 2], [-1, -1, -1, 1])

        # Calculate y, u, v channel
        _y = (0.299 * _r) + (0.587 * _g) + (0.114 * _b)
        _u = (-0.14713 * _r) - (0.28886 * _g) + (0.436 * _b)
        _v = (0.615 * _r) - (0.51499 * _g) - (0.10001 * _b)

        # Get image with YUV color space
        yuv_image = tf.concat(concat_dim=3, values=[_y, _u, _v])

        if normalize_yuv:
            # Normalize y, u, v channels
            yuv_image = normalized_yuv(yuv_image)

        return yuv_image


def yuv_to_rgb(yuv_image, scope):
    """
    Convert image color space from YUV to RGB
    :param yuv_image: an image with YUV color space
    :param scope: scope for this function
    :return: an image with RGB color space
    """
    with tf.name_scope(scope):
        if normalize_yuv:
            # Denormalize y, u, v channels
            yuv_image = denormalized_yuv(yuv_image)

        # Get y, u, v channel
        _y = tf.slice(yuv_image, [0, 0, 0, 0], [-1, -1, -1, 1])
        _u = tf.slice(yuv_image, [0, 0, 0, 1], [-1, -1, -1, 1])
        _v = tf.slice(yuv_image, [0, 0, 0, 2], [-1, -1, -1, 1])

        # Calculate r, g, b channel
        _r = (_y + 1.13983 * _v) * 255
        _g = (_y - 0.39464 * _u - 0.58060 * _v) * 255
        _b = (_y + 2.03211 * _u) * 255

        # Get image with RGB color space
        rgb_image = tf.concat(concat_dim=3, values=[_r, _g, _b])
        rgb_image = tf.maximum(rgb_image, tf.zeros(rgb_image.get_shape(), dtype=tf.float32))
        rgb_image = tf.minimum(rgb_image, tf.mul(tf.ones(rgb_image.get_shape(), dtype=tf.float32), 255))
        rgb_image = tf.div(rgb_image, 255)

        return rgb_image


def normalized_yuv(yuv_images):
    """
    Normalize the yuv image data
    :param yuv_images: the YUV images that needs normalization
    :return: the normalized yuv image
    """
    with tf.name_scope("normalize_yuv"):
        # Split channels
        channel_y = tf.slice(yuv_images, [0, 0, 0, 0], [-1, -1, -1, 1])
        channel_u = tf.slice(yuv_images, [0, 0, 0, 1], [-1, -1, -1, 1])
        channel_v = tf.slice(yuv_images, [0, 0, 0, 2], [-1, -1, -1, 1])

        # Normalize u, v channels
        channel_u = tf.div(channel_u, u_norm_para)
        channel_v = tf.div(channel_v, v_norm_para)
        channel_u = tf.add(tf.div(channel_u, 2.0), 0.5, name="channel_u")
        channel_v = tf.add(tf.div(channel_v, 2.0), 0.5, name="channel_v")

        # Add channel data
        channel_yuv = tf.concat(concat_dim=3, values=[channel_y, channel_u, channel_v], name="channel_yuv")
        return channel_yuv


def denormalized_yuv(yuv_images):
    """
    Denormalize the yuv image data
    :param yuv_images: the YUV images that needs denormalization
    :return: the denormalized yuv image
    """
    with tf.name_scope("denormalize_yuv"):
        # Split channels
        channel_y = tf.slice(yuv_images, [0, 0, 0, 0], [-1, -1, -1, 1])
        channel_u = tf.slice(yuv_images, [0, 0, 0, 1], [-1, -1, -1, 1])
        channel_v = tf.slice(yuv_images, [0, 0, 0, 2], [-1, -1, -1, 1])

        # Denormalize u, v channels
        channel_u = tf.mul(tf.sub(channel_u, 0.5), 2.0)
        channel_v = tf.mul(tf.sub(channel_v, 0.5), 2.0)
        channel_u = tf.mul(channel_u, u_norm_para, name="channel_u")
        channel_v = tf.mul(channel_v, v_norm_para, name="channel_v")

        # Add channel data
        channel_yuv = tf.concat(concat_dim=3, values=[channel_y, channel_u, channel_v], name="channel_yuv")
        return channel_yuv


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

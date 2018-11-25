#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Automatic image colorization using residual encoder network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = 'Chong Guo <armourcy@email.com>'
__copyright__ = 'Copyright 2018, Chong Guo'
__license__ = 'GPL'

import os
import cv2
import argparse
import imghdr
import numpy as np
import tensorflow as tf
from vgg import vgg16

IMAGE_SIZE = 224
IMAGE_CHANNELS = 3
U_CHANNEL_NORM = 0.435912
V_CHANNEL_NORM = 0.614777
SHUFFLE_BUFFER_SIZE = 2000


def create_folder(folder_path):
    """
    Create folder if not exist.
    :param folder_path: the folder path we want to create
    :return: None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def rgb_to_yuv(rgb_image, scope):
    """
    Convert image color space from RGB to YUV.
    :param rgb_image: an image with RGB color space
    :param scope: scope for this function
    :return: an image with YUV color space
    """
    with tf.name_scope(scope):
        # Get r, g, b channel
        _r, _g, _b = tf.split(rgb_image, axis=3, num_or_size_splits=3)

        # Calculate y, u, v channel
        # https://www.pcmag.com/encyclopedia/term/55166/yuv-rgb-conversion-formulas
        _y = 0.299 * _r + 0.587 * _g + 0.114 * _b
        _u = 0.492 * (_b - _y)
        _v = 0.877 * (_r - _y)

        # Normalize u, v channel
        _u = _u / (U_CHANNEL_NORM * 2) + 0.5
        _v = _v / (V_CHANNEL_NORM * 2) + 0.5

        # Get image with YUV color space
        yuv_image = tf.concat(axis=3, values=[_y, _u, _v])
        return tf.clip_by_value(yuv_image, 0.0, 1.0)


def yuv_to_rgb(yuv_image, scope):
    """
    Convert image color space from YUV to RGB.
    :param yuv_image: an image with YUV color space
    :param scope: scope for this function
    :return: an image with RGB color space
    """
    with tf.name_scope(scope):
        # Get y, u, v channel
        _y, _u, _v = tf.split(yuv_image, axis=3, num_or_size_splits=3)

        # Denormalize u, v channel
        _u = (_u - 0.5) * U_CHANNEL_NORM * 2
        _v = (_v - 0.5) * V_CHANNEL_NORM * 2

        # Calculate r, g, b channel
        # https://www.pcmag.com/encyclopedia/term/55166/yuv-rgb-conversion-formulas
        _r = _y + 1.14 * _v
        _g = _y - 0.395 * _u - 0.581 * _v
        _b = _y + 2.033 * _u

        # Get image with RGB color space
        rgb_image = tf.concat(axis=3, values=[_r, _g, _b])
        return tf.clip_by_value(rgb_image, 0.0, 1.0)


def init_file_path(directory, debug=False):
    """
    Get the image file path array.
    :param directory: the directory that store images
    :param debug: if enable, will delete all gray space images
    :return: an array of image file path
    """
    paths = []

    for file_name in os.listdir(directory):
        file_path = '%s/%s' % (directory, file_name)
        # Skip all non-jpg files
        if imghdr.what(file_path) is not 'jpeg':
            continue
        # Delete all gray space images (it takes time)
        if not debug:
            is_gray_space = True
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if len(img.shape) == 3 and img.shape[2] == IMAGE_CHANNELS:
                if np.any(np.not_equal(img[0], img[1])) or \
                   np.any(np.not_equal(img[1], img[2])):
                    is_gray_space = False
            if is_gray_space:
                try:
                    os.remove(file_path)
                    print("Remove gray space image file: %s" % file_name)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))
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
    rgb_image = tf.image.decode_jpeg(content, channels=IMAGE_CHANNELS, name="color_image_int")
    # Resize image to the right image size
    rgb_image_resized = tf.image.resize_images(rgb_image, [IMAGE_SIZE, IMAGE_SIZE])
    # Map all pixel element value into [0, 1]
    rgb_image_float = tf.clip_by_value(rgb_image_resized/255, 0.0, 1.0, name="color_image_float")
    return { "image": rgb_image_float }


def data_input_func(filenames, batch_size, shuffle=False):
    """
    Dataset input function which shuffle the input data and returns an iterator to get batches of images.
    :param filenames: filenames
    :param batch_size: batch size
    :param shuffle: random shuffle dataset
    :return: the batch image data iterator
    """
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle:
        dataset = dataset.apply(
            tf.data.experimental.shuffle_and_repeat(SHUFFLE_BUFFER_SIZE))
    else:
        dataset = dataset.repeat()
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            read_image, num_parallel_calls=8, batch_size=batch_size))
    return dataset


def residual_encoder(features, labels, mode, params):
    # We don't have labels for this task
    assert labels == None

    # Build residual encoder model
    print("🤖 Build residual encoder model...")
    model = ResidualEncoder()

    # Get color image
    input_layer_image = tf.feature_column.input_layer(features, params['feature_columns'])
    color_image_rgb = tf.reshape(input_layer_image, name="color_image_rgb",
                          shape=[params['batch_size'], IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
    color_image_yuv = rgb_to_yuv(color_image_rgb, "color_image_yuv")
    color_image_uv = tf.slice(color_image_yuv, [0, 0, 0, 1], [-1, -1, -1, 2], name="color_image_uv")

    # Get gray image
    gray_image_grayscale = tf.image.rgb_to_grayscale(color_image_rgb, name="gray_image_grayscale")
    gray_image_rgb = tf.image.grayscale_to_rgb(gray_image_grayscale, name="gray_image_rgb")
    gray_image_yuv = rgb_to_yuv(gray_image_rgb, "gray_image_yuv")
    gray_image_y = tf.slice(gray_image_yuv, [0, 0, 0, 0], [-1, -1, -1, 1], name="gray_image_y")

    # Build vgg model
    print("🤖 Load vgg16 model...")
    with tf.name_scope("vgg16"):
        vgg = vgg16.Vgg16()
        vgg.build(gray_image_rgb)

    # Predict model
    predict = model.build(input_data=gray_image_rgb, vgg=vgg, is_training=mode==tf.estimator.ModeKeys.TRAIN)
    predict_yuv = tf.concat(axis=3, values=[gray_image_y, predict], name="predict_yuv")
    predict_rgb = yuv_to_rgb(predict_yuv, "predict_rgb")
    compare_result = tf.concat([gray_image_rgb, predict_rgb, color_image_rgb], axis=2, name="compare_result")

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions={})

    # Get loss
    loss = model.get_loss(predict_val=predict, real_val=color_image_uv)

    # Init tensorflow summaries
    print("⏳ Init tensorflow summaries...")
    tf.summary.histogram("loss", loss)
    tf.summary.image("result", compare_result)

    # Compute evaluation metrics.
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    # Create training op.
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(name='adam_optimizer')
        train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step(), name='train_op')
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


class ResidualEncoder():
    def __init__(self):
        # Trainable weights for each conv layer
        self.trainable_weights = {
            'b_conv4': tf.get_variable('b_conv4', trainable=True,
                           initializer=tf.truncated_normal([1, 1, 512, 256], stddev=0.01)),
            'b_conv3': tf.get_variable('b_conv3', trainable=True,
                           initializer=tf.truncated_normal([3, 3, 256, 128], stddev=0.01)),
            'b_conv2': tf.get_variable('b_conv2', trainable=True,
                           initializer=tf.truncated_normal([3, 3, 128, 64], stddev=0.01)),
            'b_conv1': tf.get_variable('b_conv1', trainable=True,
                           initializer=tf.truncated_normal([3, 3, 64, 3], stddev=0.01)),
            'b_conv0': tf.get_variable('b_conv0', trainable=True,
                           initializer=tf.truncated_normal([3, 3, 3, 3], stddev=0.01)),
            'output_conv': tf.get_variable('output_conv', trainable=True,
                               initializer=tf.truncated_normal([3, 3, 3, 2], stddev=0.01)),
        }

        # Gaussian blur kernel (not trainable)
        self.blur_3x3 = np.array([
            [1., 2., 1.],
            [2., 4., 2.],
            [1., 2., 1.],
        ]) / 16 # (3, 3)
        self.blur_3x3 = np.stack((self.blur_3x3, self.blur_3x3), axis=-1) # (3, 3, 2)
        self.blur_3x3 = np.stack((self.blur_3x3, self.blur_3x3), axis=-1) # (3, 3, 2, 2)
        self.blur_3x3 = tf.get_variable('blur_3x3', trainable=False,
                            initializer=tf.convert_to_tensor(self.blur_3x3, dtype=tf.float32))

        self.blur_5x5 = np.array([
            [1.,  4.,  7.,  4., 1.],
            [4., 16., 26., 16., 4.],
            [7., 26., 41., 26., 7.],
            [4., 16., 26., 16., 4.],
            [1.,  4.,  7.,  4., 1.],
        ]) / 273 # (5, 5)
        self.blur_5x5 = np.stack((self.blur_5x5, self.blur_5x5), axis=-1) # (5, 5, 2)
        self.blur_5x5 = np.stack((self.blur_5x5, self.blur_5x5), axis=-1) # (5, 5, 2, 2)
        self.blur_5x5 = tf.get_variable('blur_5x5', trainable=False,
                            initializer=tf.convert_to_tensor(self.blur_5x5, dtype=tf.float32))

    def get_loss(self, predict_val, real_val, debug=False):
        """
        Loss function.
        :param predict_val: the predict value
        :param real_val: the real value
        :param debug: debug mode
        :return: loss
        """
        if debug:
            assert predict_val.get_shape().as_list()[1:] == [224, 224, 2]
            assert real_val.get_shape().as_list()[1:] == [224, 224, 2]

        blur_real_3x3 = tf.nn.conv2d(real_val, self.blur_3x3,
                            strides=[1, 1, 1, 1], padding='SAME', name="blur_real_3x3")
        blur_real_5x5 = tf.nn.conv2d(real_val, self.blur_5x5,
                            strides=[1, 1, 1, 1], padding='SAME', name="blur_real_5x5")
        blur_predict_3x3 = tf.nn.conv2d(predict_val, self.blur_3x3,
                               strides=[1, 1, 1, 1], padding='SAME', name="blur_predict_3x3")
        blur_predict_5x5 = tf.nn.conv2d(predict_val, self.blur_5x5,
                               strides=[1, 1, 1, 1], padding='SAME', name="blur_predict_5x5")

        diff_original = tf.reduce_sum(tf.squared_difference(predict_val, real_val),
                            name="diff_original")
        diff_blur_3x3 = tf.reduce_sum(tf.squared_difference(blur_predict_3x3, blur_real_3x3),
                            name="diff_blur_3x3")
        diff_blur_5x5 = tf.reduce_sum(tf.squared_difference(blur_predict_5x5, blur_real_5x5),
                           name="diff_blur_5x5")
        return (diff_original + diff_blur_3x3 + diff_blur_5x5) / 3

    def batch_normal(self, input_data, scope, is_training):
        """
        Batch normalization using build-in batch_normalization function.
        :param input_data: the input data
        :param scope: scope
        :param is_training: the flag indicate if it is training
        :return: normalized data
        """
        return tf.layers.batch_normalization(input_data, training=is_training, name=scope)

    def conv_layer(self, layer_input, scope, is_training, relu=True, bn=True):
        """
        Convolution layer.
        :param layer_input: the input data for this layer
        :param scope: scope for this layer
        :param is_training: a flag indicate if now is in training
        :param relu: relu flag
        :param bn: batch normalize flag
        :return: the layer data after convolution
        """
        with tf.name_scope(scope):
            output = tf.nn.conv2d(layer_input, self.trainable_weights[scope],
                         strides=[1, 1, 1, 1], padding='SAME', name="conv")
            if bn:
                output = self.batch_normal(output, is_training=is_training, scope=scope + '_bn')
            if relu:
                output = tf.nn.relu(output, name="relu")
            else:
                output = tf.sigmoid(output, name="sigmoid")
            return output

    def build(self, input_data, vgg, is_training, debug=False):
        """
        Build the residual encoder model.
        :param input_data: input data for first layer
        :param vgg: the vgg model
        :param is_training: a flag indicate if now is in training
        :param debug: debug mode
        :return: None
        """
        if debug:
            assert input_data.get_shape().as_list()[1:] == [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS]

        # Batch norm and 1x1 convolutional layer 4
        bn_4 = self.batch_normal(vgg.conv4_3, "bn_4", is_training)
        b_conv4 = self.conv_layer(bn_4, "b_conv4", is_training, bn=False)

        if debug:
            assert bn_4.get_shape().as_list()[1:] == [28, 28, 512]
            assert b_conv4.get_shape().as_list()[1:] == [28, 28, 256]

        # Backward upscale layer 4 and add convolutional layer 3
        b_conv4_upscale = tf.image.resize_images(b_conv4, [56, 56])
        bn_3 = self.batch_normal(vgg.conv3_3, "bn_3", is_training)
        b_conv3_input = tf.add(bn_3, b_conv4_upscale, name="b_conv3_input")
        b_conv3 = self.conv_layer(b_conv3_input, "b_conv3", is_training)

        if debug:
            assert b_conv4_upscale.get_shape().as_list()[1:] == [56, 56, 256]
            assert bn_3.get_shape().as_list()[1:] == [56, 56, 256]
            assert b_conv3_input.get_shape().as_list()[1:] == [56, 56, 256]
            assert b_conv3.get_shape().as_list()[1:] == [56, 56, 128]

        # Backward upscale layer 3 and add convolutional layer 2
        b_conv3_upscale = tf.image.resize_images(b_conv3, [112, 112])
        bn_2 = self.batch_normal(vgg.conv2_2, "bn_2", is_training)
        b_conv2_input = tf.add(bn_2, b_conv3_upscale, name="b_conv2_input")
        b_conv2 = self.conv_layer(b_conv2_input, "b_conv2", is_training)

        if debug:
            assert b_conv3_upscale.get_shape().as_list()[1:] == [112, 112, 128]
            assert bn_2.get_shape().as_list()[1:] == [112, 112, 128]
            assert b_conv2_input.get_shape().as_list()[1:] == [112, 112, 128]
            assert b_conv2.get_shape().as_list()[1:] == [112, 112, 64]

        # Backward upscale layer 2 and add convolutional layer 1
        b_conv2_upscale = tf.image.resize_images(b_conv2, [224, 224])
        bn_1 = self.batch_normal(vgg.conv1_2, "bn_1", is_training)
        b_conv1_input = tf.add(bn_1, b_conv2_upscale, name="b_conv1_input")
        b_conv1 = self.conv_layer(b_conv1_input, "b_conv1", is_training)

        if debug:
            assert b_conv2_upscale.get_shape().as_list()[1:] == [224, 224, 64]
            assert bn_1.get_shape().as_list()[1:] == [224, 224, 64]
            assert b_conv1_input.get_shape().as_list()[1:] == [224, 224, 64]
            assert b_conv1.get_shape().as_list()[1:] == [224, 224, 3]

        # Backward upscale layer 1 and add input layer
        bn_0 = self.batch_normal(input_data, "bn_0", is_training)
        b_conv0_input = tf.add(bn_0, b_conv1, name="b_conv0_input")
        b_conv0 = self.conv_layer(b_conv0_input, "b_conv0", is_training)

        if debug:
            assert bn_0.get_shape().as_list()[1:] == [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS]
            assert b_conv0_input.get_shape().as_list()[1:] == [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS]
            assert b_conv0.get_shape().as_list()[1:] == [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS]

        # Output layer
        output_layer = self.conv_layer(b_conv0, "output_conv", is_training, relu=False)

        if debug:
            assert output_layer.get_shape().as_list()[1:] == [IMAGE_SIZE, IMAGE_SIZE, 2]

        return output_layer


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False, action='store_true',
                        help='if enable debug mode will run all the assertions')
    parser.add_argument('--batch_size', default=6, type=int,
                        help='number of batch size')
    parser.add_argument('--train_steps', default=1000000, type=int,
                        help='number of training iterations')
    parser.add_argument('--save_summary_steps', default=500, type=int,
                        help='number of steps to save summaries')
    parser.add_argument('--save_model_steps', default=10000, type=int,
                        help='number of steps to save model')
    parser.add_argument('--training_dir', default="train2014", type=str,
                        help='directory for training dataset')
    parser.add_argument('--validating_dir', default="val2014", type=str,
                        help='directory for validating dataset')
    parser.add_argument('--testing_dir', default="test2014", type=str,
                        help='directory for testing dataset')
    parser.add_argument('--summary_dir', default="summary", type=str,
                        help='directory for tensorflow summaries')
    args = parser.parse_args(argv[1:])

    # Create summary folder if not exist
    create_folder(args.summary_dir + "/train/images")
    create_folder(args.summary_dir + "/test/images")

    # Feature columns describe how to use the input
    feature_columns = [tf.feature_column.numeric_column('image',
                           shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))]

    # Create run_config to add GPU support
    distribution = tf.contrib.distribute.MirroredStrategy()
    gpu_options = tf.GPUOptions(allow_growth=True)
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=True,
                                    gpu_options=gpu_options)
    run_config = tf.estimator.RunConfig(session_config=session_config,
                                        train_distribute=distribution,
                                        save_checkpoints_steps=args.save_model_steps,
                                        save_summary_steps=args.save_summary_steps)

    # Create custom estimator
    classifier = tf.estimator.Estimator(
        model_fn=residual_encoder,
        model_dir=args.summary_dir,
        config=run_config,
        params={
            'feature_columns': feature_columns,
            'batch_size': args.batch_size,
        }
    )

    # Train model
    print("🤖 Start training...")
    classifier.train(
        input_fn=lambda:data_input_func(init_file_path(args.training_dir, debug=args.debug),
                                        batch_size=args.batch_size, shuffle=True),
        steps=args.train_steps)
    print("🎉 Training finished!")

    # Evaluate model
    print("🤖 Start testing...")
    eval_result = classifier.evaluate(
        input_fn=lambda:data_input_func(init_file_path(args.testing_dir, debug=args.debug),
                                        batch_size=args.batch_size, shuffle=False))
    print("🎉 Testing finished!")
    print('👀 Test set loss: {loss:0.4f}\n'.format(**eval_result))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    np.random.seed(42)
    tf.set_random_seed(42)
    tf.app.run(main)

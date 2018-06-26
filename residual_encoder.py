#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Residual-encoder model implementation.

See extensive documentation at
http://tinyclouds.org/colorize/
"""

import cv2
import tensorflow as tf

from config import batch_size, debug, weights, training_resize_method, tf_blur_3x3, tf_blur_5x5


class ResidualEncoder(object):
    @staticmethod
    def get_weight(scope):
        """
        Get initial weight for specific layer
        :param scope: the scope of the layer
        :return: the initial weight for this layer
        """
        return weights[scope]

    @staticmethod
    def get_loss(predict_val, real_val):
        """
        Loss function
        :param predict_val: the predict value
        :param real_val: the real value
        :return: cost
        """
        if debug:
            assert predict_val.get_shape().as_list()[1:] == [224, 224, 2]
            assert real_val.get_shape().as_list()[1:] == [224, 224, 2]

        blur_real_3x3 = tf.nn.conv2d(real_val, tf_blur_3x3, strides=[1, 1, 1, 1], padding='SAME', name="blur_real_3x3")
        blur_real_5x5 = tf.nn.conv2d(real_val, tf_blur_5x5, strides=[1, 1, 1, 1], padding='SAME', name="blur_real_5x5")
        blur_predict_3x3 = tf.nn.conv2d(predict_val, tf_blur_3x3, strides=[1, 1, 1, 1], padding='SAME', name="blur_predict_3x3")
        blur_predict_5x5 = tf.nn.conv2d(predict_val, tf_blur_5x5, strides=[1, 1, 1, 1], padding='SAME', name="blur_predict_5x5")

        diff_original = tf.abs(tf.subtract(predict_val, real_val), name="diff_original")
        diff_blur_3x3 = tf.abs(tf.subtract(blur_predict_3x3, blur_real_3x3), name="diff_blur_3x3")
        diff_blur_5x5 = tf.abs(tf.subtract(blur_predict_5x5, blur_real_5x5), name="diff_blur_5x5")
        return tf.div(tf.add_n([diff_original, diff_blur_3x3, diff_blur_5x5]), 3)

    @staticmethod
    def batch_normal(input_data, scope, training_flag):
        """
        Batch normalization with build-in batch_normalization function
        :param input_data: the input data
        :param scope: scope
        :param training_flag: the flag indicate if it is training
        :return: normalized data
        """
        return tf.layers.batch_normalization(input_data, training=training_flag, name=scope)

    def conv_layer(self, layer_input, scope, is_training, relu=True, bn=True):
        """
        Convolution layer
        :param layer_input: the input data for this layer
        :param scope: scope for this layer
        :param is_training: a flag indicate if now is in training
        :param relu: relu flag
        :param bn: batch normalize flag
        :return: the layer data after convolution
        """
        with tf.variable_scope(scope):
            weight = self.get_weight(scope)
            output = tf.nn.conv2d(layer_input, weight, strides=[1, 1, 1, 1], padding='SAME', name="conv")
            if bn:
                output = self.batch_normal(output, training_flag=is_training, scope=scope + '_bn')
            if relu:
                output = tf.nn.relu(output, name="relu")
            else:
                output = tf.sigmoid(output, name="sigmoid")
            return output

    def build(self, input_data, vgg, is_training):
        """
        Build the residual encoder model
        :param input_data: input data for first layer
        :param vgg: the vgg model
        :param is_training: a flag indicate if now is in training
        :return: None
        """
        if debug:
            assert input_data.get_shape().as_list()[1:] == [224, 224, 3]

        # Batch norm and 1x1 convolutional layer 4
        bn_4 = self.batch_normal(vgg.conv4_3, "bn_4", is_training)
        b_conv4 = self.conv_layer(bn_4, "b_conv4", is_training, bn=False)

        if debug:
            assert bn_4.get_shape().as_list()[1:] == [28, 28, 512]
            assert b_conv4.get_shape().as_list()[1:] == [28, 28, 256]

        # Backward upscale layer 4 and add convolutional layer 3
        b_conv4_upscale = tf.image.resize_images(b_conv4, [56, 56], method=training_resize_method)
        bn_3 = self.batch_normal(vgg.conv3_3, "bn_3", is_training)
        b_conv3_input = tf.add(bn_3, b_conv4_upscale, name="b_conv3_input")
        b_conv3 = self.conv_layer(b_conv3_input, "b_conv3", is_training)

        if debug:
            assert b_conv4_upscale.get_shape().as_list()[1:] == [56, 56, 256]
            assert bn_3.get_shape().as_list()[1:] == [56, 56, 256]
            assert b_conv3_input.get_shape().as_list()[1:] == [56, 56, 256]
            assert b_conv3.get_shape().as_list()[1:] == [56, 56, 128]

        # Backward upscale layer 3 and add convolutional layer 2
        b_conv3_upscale = tf.image.resize_images(b_conv3, [112, 112], method=training_resize_method)
        bn_2 = self.batch_normal(vgg.conv2_2, "bn_2", is_training)
        b_conv2_input = tf.add(bn_2, b_conv3_upscale, name="b_conv2_input")
        b_conv2 = self.conv_layer(b_conv2_input, "b_conv2", is_training)

        if debug:
            assert b_conv3_upscale.get_shape().as_list()[1:] == [112, 112, 128]
            assert bn_2.get_shape().as_list()[1:] == [112, 112, 128]
            assert b_conv2_input.get_shape().as_list()[1:] == [112, 112, 128]
            assert b_conv2.get_shape().as_list()[1:] == [112, 112, 64]

        # Backward upscale layer 2 and add convolutional layer 1
        b_conv2_upscale = tf.image.resize_images(b_conv2, [224, 224], method=training_resize_method)
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
            assert bn_0.get_shape().as_list()[1:] == [224, 224, 3]
            assert b_conv0_input.get_shape().as_list()[1:] == [224, 224, 3]
            assert b_conv0.get_shape().as_list()[1:] == [224, 224, 3]

        # Output layer
        output_layer = self.conv_layer(b_conv0, "output_conv", is_training, relu=False)

        if debug:
            assert output_layer.get_shape().as_list()[1:] == [224, 224, 2]

        return output_layer

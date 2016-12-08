"""
Residual-encoder model implementation.

See extensive documentation at
http://tinyclouds.org/colorize/
"""

from tensorflow.contrib.layers import batch_norm

from config import *
from batchnorm import ConvolutionalBatchNormalizer


class ResidualEncoder(object):
    def __init__(self):
        pass

    @staticmethod
    def get_weight(scope):
        """
        Get weight for one layer
        :param scope: the scope of the layer
        :return: the initial weight for this layer
        """
        return weights[scope]

    @staticmethod
    def get_cost(predict_val, real_val):
        """
        Cost function
        :param predict_val: the predict value
        :param real_val: the real value
        :return: cost
        """
        if debug:
            assert predict_val.get_shape().as_list()[1:] == [224, 224, 2]
            assert real_val.get_shape().as_list()[1:] == [224, 224, 2]

        diff = tf.sub(predict_val, real_val, name="diff")
        square = tf.square(diff, name="square")
        return square

    @staticmethod
    def batch_normal(input_data, scope, training_flag, depth):
        """
        Doing batch normalization
        :param input_data: the input data
        :param scope: scope
        :param training_flag: the flag indicate if it is training
        :param depth: depth for batch normalizer
        :return: normalized data
        """
        with tf.variable_scope(scope):
            ewma = tf.train.ExponentialMovingAverage(decay=0.9999)
            bn = ConvolutionalBatchNormalizer(depth, 0.001, ewma, True)
            update_assignments = bn.get_assigner()
            x = bn.normalize(input_data, train=training_flag)
        return x

    @staticmethod
    def batch_normal_new(input_data, scope, training_flag):
        """
        Doing batch normalization, this is the new version with build-in batch_norm function
        :param input_data: the input data
        :param scope: scope
        :param training_flag: the flag indicate if it is training
        :return: normalized data
        """
        return tf.cond(training_flag,
                       lambda: batch_norm(input_data, decay=0.9999, is_training=True, center=True, scale=True,
                                          updates_collections=None, scope=scope),
                       lambda: batch_norm(input_data, decay=0.9999, is_training=False, center=True, scale=True,
                                          updates_collections=None, scope=scope, reuse=True),
                       name='batch_normalization')

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
                output = self.batch_normal(output, training_flag=is_training, scope=scope, depth=weight.get_shape()[3])
            if relu:
                output = tf.nn.relu(output, name="relu")
            else:
                output = tf.tanh(output, name="tanh")
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
        bn_4 = self.batch_normal(vgg.conv4_3, "b_conv4", is_training, 512)
        b_conv4 = self.conv_layer(bn_4, "b_conv4", is_training, bn=False)

        if debug:
            assert bn_4.get_shape().as_list()[1:] == [28, 28, 512]
            assert b_conv4.get_shape().as_list()[1:] == [28, 28, 256]

        # Backward upscale layer 4 and add convolutional layer 3
        b_conv4_upscale = tf.image.resize_images(b_conv4, [56, 56], method=training_resize_method)
        bn_3 = self.batch_normal(vgg.conv3_3, "b_conv3", is_training, 256)
        b_conv3_input = tf.add(bn_3, b_conv4_upscale, name="b_conv3_input")
        b_conv3 = self.conv_layer(b_conv3_input, "b_conv3", is_training)

        if debug:
            assert b_conv4_upscale.get_shape().as_list()[1:] == [56, 56, 256]
            assert bn_3.get_shape().as_list()[1:] == [56, 56, 256]
            assert b_conv3_input.get_shape().as_list()[1:] == [56, 56, 256]
            assert b_conv3.get_shape().as_list()[1:] == [56, 56, 128]

        # Backward upscale layer 3 and add convolutional layer 2
        b_conv3_upscale = tf.image.resize_images(b_conv3, [112, 112], method=training_resize_method)
        bn_2 = self.batch_normal(vgg.conv2_2, "b_conv2", is_training, 128)
        b_conv2_input = tf.add(bn_2, b_conv3_upscale, name="b_conv2_input")
        b_conv2 = self.conv_layer(b_conv2_input, "b_conv2", is_training)

        if debug:
            assert b_conv3_upscale.get_shape().as_list()[1:] == [112, 112, 128]
            assert bn_2.get_shape().as_list()[1:] == [112, 112, 128]
            assert b_conv2_input.get_shape().as_list()[1:] == [112, 112, 128]
            assert b_conv2.get_shape().as_list()[1:] == [112, 112, 64]

        # Backward upscale layer 2 and add convolutional layer 1
        b_conv2_upscale = tf.image.resize_images(b_conv2, [224, 224], method=training_resize_method)
        bn_1 = self.batch_normal(vgg.conv1_2, "b_conv1", is_training, 64)
        b_conv1_input = tf.add(bn_1, b_conv2_upscale, name="b_conv1_input")
        b_conv1 = self.conv_layer(b_conv1_input, "b_conv1", is_training)

        if debug:
            assert b_conv2_upscale.get_shape().as_list()[1:] == [224, 224, 64]
            assert bn_1.get_shape().as_list()[1:] == [224, 224, 64]
            assert b_conv1_input.get_shape().as_list()[1:] == [224, 224, 64]
            assert b_conv1.get_shape().as_list()[1:] == [224, 224, 3]

        # Backward upscale layer 1 and add input layer
        bn_0 = self.batch_normal(input_data, "b_conv0", is_training, 3)
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

"""
The common structure for training and testing
"""

import os

from config import *
from image_helper import (rgb_to_yuv, yuv_to_rgb)
from read_input import (init_file_path, input_pipeline)
from residual_encoder import ResidualEncoder
from vgg import vgg16


def create_folder(folder_path):
    """
    Create folder if not exist
    :param folder_path:
    :return: None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def init_model(train=True):
    """
    Init model for both training and testing
    :param train: indicate if current is in training
    :return: all stuffs that need for this model
    """
    # Create training summary folder if not exist
    create_folder("summary/train/images")

    # Create testing summary folder if not exist
    create_folder("summary/test/images")

    # Init image data file path
    print "Init file path"
    if train:
        file_paths = init_file_path(train_dir)
    else:
        file_paths = init_file_path(test_dir)

    # Init placeholder and global step
    print "Init placeholder"
    is_training = tf.placeholder(tf.bool, name="training_flag")
    global_step = tf.Variable(0, name='global_step', trainable=False)
    uv = tf.placeholder(tf.uint8, name='uv')

    # Init vgg16 model
    print "Init vgg16 model"
    vgg = vgg16.Vgg16()

    # Init residual encoder model
    print "Init residual encoder model"
    residual_encoder = ResidualEncoder()

    # Color image
    color_image_rgb = input_pipeline(file_paths, batch_size)
    color_image_yuv = rgb_to_yuv(color_image_rgb, "rgb2yuv_for_color_image")

    # Gray image
    gray_image = tf.image.rgb_to_grayscale(color_image_rgb, name="gray_image")
    gray_image_rgb = tf.image.grayscale_to_rgb(gray_image, name="gray_image_rgb")
    gray_image_yuv = rgb_to_yuv(gray_image_rgb, "rgb2yuv_for_gray_image")
    gray_image = tf.concat(concat_dim=3, values=[gray_image, gray_image, gray_image], name="gray_image_input")

    # Build vgg model
    with tf.name_scope("content_vgg"):
        vgg.build(gray_image)

    # Predict model
    predict = residual_encoder.build(input_data=gray_image, vgg=vgg, is_training=is_training)
    predict_yuv = tf.concat(concat_dim=3, values=[tf.slice(gray_image_yuv, [0, 0, 0, 0], [-1, -1, -1, 1], name="gray_image_y"), predict], name="predict_yuv")
    predict_rgb = yuv_to_rgb(predict_yuv, "yuv2rgb_for_pred_image")

    # Cost
    cost = residual_encoder.get_cost(predict_val=predict, real_val=tf.slice(color_image_yuv, [0, 0, 0, 1], [-1, -1, -1, 2], name="color_image_uv"))

    u_channel_cost = tf.slice(cost, [0, 0, 0, 0], [-1, -1, -1, 1], name="u_channel_cost")
    v_channel_cost = tf.slice(cost, [0, 0, 0, 1], [-1, -1, -1, 1], name="v_channel_cost")

    cost = tf.case({tf.equal(uv, 1): lambda: u_channel_cost,
                    tf.equal(uv, 2): lambda: v_channel_cost},
                   default=lambda: (u_channel_cost + v_channel_cost) / 2,
                   exclusive=True, name="cost")

    # Using different learning rate in different training steps
    lr = tf.div(learning_rate, tf.cast(tf.pow(2, tf.div(global_step, 8000)), tf.float32), name="learning_rate")

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost, global_step=global_step)

    # Summaries
    print "Init summaries"
    tf.histogram_summary("cost", tf.reduce_mean(cost))
    tf.image_summary("color_image_rgb", color_image_rgb, max_images=1)
    tf.image_summary("predict_rgb", predict_rgb, max_images=1)
    tf.image_summary("gray_image", gray_image_rgb, max_images=1)

    return is_training, global_step, uv, optimizer, cost, predict, predict_rgb, color_image_rgb, gray_image_rgb, file_paths

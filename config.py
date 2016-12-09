"""
Config file
"""

import tensorflow as tf
from tensorflow.python.ops.image_ops import ResizeMethod


# Debug flag, if true, will check model shape using assert in each step and skip gray image check part (to save time)
debug = True

# Image size for training
image_size = 224

# Parameters for neural network
learning_rate = 1e-6  # Initial learning rate, every 5000 step we divide this by 2
training_iters = 80000  # The training iterations number
batch_size = 30  # The batch size
display_step = 1  # Display loss for each step
test_step = 1000  # Test and save image during training phase
save_step = 5000  # Save our model
dequeue_buffer_size = 1000

# Image resize method
input_resize_method = ResizeMethod.BILINEAR
training_resize_method = ResizeMethod.BILINEAR

# YUV normalization parameters
normalize_yuv = True
y_norm_para = 0.5
u_norm_para = 0.436
v_norm_para = 0.615

# Directory for training and testing dataset
train_dir = "train2014"
test_dir = "val2014"

# Summary directory for training and testing
train_summary = "summary/train"
test_summary = "summary/test"

# Model and generated images stored path
model_path = "summary"

# Weights for each layer
weights = {
    'b_conv4': tf.Variable(tf.truncated_normal([1, 1, 512, 256], stddev=0.01)),
    'b_conv3': tf.Variable(tf.truncated_normal([3, 3, 256, 128], stddev=0.01)),
    'b_conv2': tf.Variable(tf.truncated_normal([3, 3, 128, 64], stddev=0.01)),
    'b_conv1': tf.Variable(tf.truncated_normal([3, 3, 64, 3], stddev=0.01)),
    'b_conv0': tf.Variable(tf.truncated_normal([3, 3, 3, 3], stddev=0.01)),
    'output_conv': tf.Variable(tf.truncated_normal([3, 3, 3, 2], stddev=0.01)),
}

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Train model."""

__author__ = 'Chong Guo'
__copyright__ = 'Copyright 2018, Chong Guo'
__license__ = 'GPL'
__email__ = 'armourcy@email.com'

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from config import display_step, summary_path, saving_step, testing_step, training_iters, training_summary
from common import init_model
from image_helper import concat_images


if __name__ == '__main__':
    # Init model
    is_training, global_step, optimizer, loss, predict_rgb, color_image_rgb, gray_image, _ = init_model(train=True)

    # Init scaffold, hooks and config
    scaffold = tf.train.Scaffold()
    summary_hook = tf.train.SummarySaverHook(output_dir=training_summary, save_steps=display_step, scaffold=scaffold)
    checkpoint_hook = tf.train.CheckpointSaverHook(checkpoint_dir=summary_path, save_steps=saving_step, scaffold=scaffold)
    num_step_hook = tf.train.StopAtStepHook(num_steps=training_iters)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, gpu_options=(tf.GPUOptions(allow_growth=True)))

    # Create a session for running operations in the Graph
    with tf.train.MonitoredTrainingSession(checkpoint_dir=summary_path,
                                           hooks=[summary_hook, checkpoint_hook, num_step_hook],
                                           scaffold=scaffold,
                                           config=config) as sess:
        print("🤖 Start training...")

        while not sess.should_stop():
            # Run optimizer
            _, step, l, pred, color, gray = sess.run([optimizer, global_step, loss, predict_rgb, color_image_rgb, gray_image] , feed_dict={is_training: True})

            if step % display_step == 0:
                # Print batch loss
                print("📖 Iter %d, Minibatch Loss = %f" % (step, l))

                # Save testing image
                if step % testing_step == 0:
                    summary_image = concat_images(gray[0], pred[0])
                    summary_image = concat_images(summary_image, color[0])
                    plt.imsave("%s/images/%d.png" % (training_summary, step), summary_image)

        print("🎉 Training finished!")

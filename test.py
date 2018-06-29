#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test model."""

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from config import batch_size, display_step, saving_step, summary_path, testing_summary
from common import init_model
from image_helper import concat_images


if __name__ == '__main__':
    # Init model.
    is_training, global_step, _, loss, predict_rgb, color_image_rgb, gray_image, file_paths = init_model(train=False)

    # Init scaffold, hooks and config.
    scaffold = tf.train.Scaffold()
    summary_hook = tf.train.SummarySaverHook(output_dir=testing_summary, save_steps=display_step, scaffold=scaffold)
    checkpoint_hook = tf.train.CheckpointSaverHook(checkpoint_dir=summary_path, save_steps=saving_step, scaffold=scaffold)
    num_step_hook = tf.train.StopAtStepHook(num_steps=len(file_paths))
    config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    session_creator = tf.train.ChiefSessionCreator(scaffold=scaffold, config=config, checkpoint_dir=summary_path)

    # Create a session for running operations in the Graph.
    with tf.train.MonitoredSession(session_creator=session_creator, hooks=[checkpoint_hook, summary_hook]) as sess:
        print("🤖 Start testing...")
        avg_loss = 0

        while not sess.should_stop():
            # Get global_step.
            step, l, pred, color, gray = sess.run([global_step, loss, predict_rgb, color_image_rgb, gray_image], feed_dict={is_training: False})

            if step % display_step == 0:
                # Print batch loss.
                print("📖 Iter %d, Minibatch Loss = %f" % (step, l))
                avg_loss += float(l)

                # Save testing image.
                summary_image = concat_images(gray[0], pred[0])
                summary_image = concat_images(summary_image, color[0])
                plt.imsave("%s/images/%d.png" % (testing_summary, step), summary_image)

        print("🎉 Testing finished!")
        print("👀 Total average loss: %f" % (avg_loss / len(file_paths)))

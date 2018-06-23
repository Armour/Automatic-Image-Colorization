#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Train model."""

import sys

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from config import batch_size, display_step, model_path, save_step, test_step, training_iters, train_summary
from common import init_model
from image_helper import concat_images


if __name__ == '__main__':
    # Init model
    is_training, global_step, optimizer, cost, predict, predict_rgb, color_image_rgb, gray_image, file_paths = init_model(train=True)

    # Saver
    print("Init model saver")
    saver = tf.train.Saver()

    # Init the graph
    print("Init graph")
    init = tf.global_variables_initializer()

    # Create a session for running operations in the Graph
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        # Initialize the variables
        sess.run(init)

        # Merge all summaries
        print("Merge all summaries")
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(train_summary, sess.graph)

        # Start input enqueue threads
        print("Start input enqueue threads")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Start training
        print("Start training!!!")

        try:
            while not coord.should_stop():
                # Update global_step
                step = tf.train.global_step(sess, global_step)

                # Run optimizer
                sess.run(optimizer, feed_dict={is_training: True})

                # Print batch loss
                if step % display_step == 0:
                    loss, pred, color, gray, summary = sess.run([cost, predict_rgb, color_image_rgb, gray_image, merged], feed_dict={is_training: False})
                    print("Iter %d, Minibatch Loss = %f" % (step, float(np.mean(loss) / batch_size)))
                    train_writer.add_summary(summary, step)
                    train_writer.flush()

                    # Save test image
                    if step % test_step == 0:
                        summary_image = concat_images(gray[0], pred[0])
                        summary_image = concat_images(summary_image, color[0])
                        plt.imsave("%s/images/%s.png" % (train_summary, str(step)), summary_image)

                # Save model
                if step % save_step == 0 and step != 0:
                    save_path = saver.save(sess, "%s/model.ckpt" % model_path)
                    print("Model saved in file: %s" % save_path)

                # Stop training
                if step == training_iters:
                    break

            print("Training Finished!")
            sys.stdout.flush()

        except tf.errors.OUT_OF_RANGE as e:
            # Handle exception
            print("Done training -- epoch limit reached")

        finally:
            # When done, ask the threads to stop
            coord.request_stop()

    # Wait for threads to finish
    coord.join(threads)
    sess.close()

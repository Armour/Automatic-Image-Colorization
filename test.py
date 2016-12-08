"""
Test model
"""

import sys

import numpy as np
from matplotlib import pyplot as plt

from config import *
from common import init_model
from image_helper import (concat_images)


if __name__ == '__main__':
    # Init model
    is_training, global_step, uv, optimizer, cost, predict, predict_rgb, color_image_rgb, gray_image_rgb, file_paths = init_model(train=False)

    # Saver
    print "Init model saver"
    saver = tf.train.Saver()

    # Init the graph
    print "Init graph"
    init = tf.initialize_all_variables()

    # Create a session for running operations in the Graph
    with tf.Session() as sess:
        # Initialize the variables.
        sess.run(init)

        # Merge all summaries
        print "Merge all summaries"
        merged = tf.merge_all_summaries()
        test_writer = tf.train.SummaryWriter(test_summary)

        # Start input enqueue threads.
        print "Start input enqueue threads"
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Load model
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Load model finished!"
        else:
            print "Failed to restore model"
            exit()

        # Start testing
        print "Start testing!!!"

        try:
            step = 0
            avg_error = 0
            while not coord.should_stop():
                step += 1

                # Print batch loss
                if step % display_step == 0:
                    loss, pred, pred_rgb, color_rgb, gray_rgb, summary = \
                        sess.run([cost, predict, predict_rgb, color_image_rgb, gray_image_rgb, merged], feed_dict={is_training: False, uv: 3})
                    print "Iter %d, Minibatch Loss = %f" % (step, float(np.mean(loss)))
                    avg_error += float(np.mean(loss))
                    test_writer.add_summary(summary, step)
                    test_writer.flush()

                    # Save output image
                    summary_image = concat_images(gray_rgb[0], pred_rgb[0])
                    summary_image = concat_images(summary_image, color_rgb[0])
                    plt.imsave("%s/images/%s.jpg" % (test_summary, str(step)), summary_image)

                if step == len(file_paths):
                    break

            print "Testing Finished!"
            print "Average error: %f" % (avg_error / len(file_paths))
            sys.stdout.flush()

        except tf.errors.OUT_OF_RANGE as e:
            # Handle exception
            print "Done training -- epoch limit reached"
            coord.request_stop(e)

        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

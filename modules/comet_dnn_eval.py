# These scripts are based on the example of TensorFlow CIFAR-10
# They use the Apache license.  We will use the same.
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
# CURRENT STATUS: eval_once() has been ignored in favor of evaluate().
# evaluate() is able to properly(?) restore a model from checkpoint (using either Saver or SavedModel),
# and then makes a few useful plots for model assessment.
# While restored variables are identical to the ones saved in checkpoints,
# histograms still look different from tensorboard by eye--reasons unclear.
# -- 2018/08/01, Jordan Xiao

"""Evaluation for comet_dnn.

Accuracy:

Speed:

Usage:
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

import comet_dnn_input
import comet_dnn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('eval_test', True,
                           """If true, evaluates the testing data""")
tf.app.flags.DEFINE_string('model_path', None,
                           """Directory to the check point after training""")
tf.app.flags.DEFINE_string('saver_model_path', None,
                           """Directory where to read checkpoints (for TF Saver).""")
tf.app.flags.DEFINE_string('saved_model_dir', None,
                           """Directory where to read saved model (for TF SavedModel).""")
tf.app.flags.DEFINE_boolean("debug_mode", False,
                            """Whether to active debug mode""")
tf.app.flags.DEFINE_boolean('show_eval_plots', True,
                           """If true, show histograms and correlation plots.""")


def eval_plots(compiled_true_labels, compiled_predicted_labels, ckpt_num):
    # made this a separate function to keep evaluate() from being too cluttered
    # INPUTS: both have dimensions of batch_size rows by num_classes columns (ckpt_num for title purposes)
    # OUTPUTS: five plots per label
    
    n = 5 # number of plots; used to prevent figure overlapping, also makes it easier to add new plots
    # as long as n is at least as large as number of plots coded for, then no overlap issues

    # Get residuals (same dimensions as label arrays)
    compiled_residuals = compiled_true_labels - compiled_predicted_labels

    for i in np.arange(FLAGS.num_classes): # i is index for cycling through labels
        lbl_name = comet_dnn_input.LABEL_NAMES[i]
        true_labels = compiled_true_labels[:,i]
        predicted_labels = compiled_predicted_labels[:,i]
        residuals = compiled_residuals[:,i]
        
        # Histograms
        plt.figure(i,figsize=(20,10))
        plt.subplot(231)
        plt.hist(true_labels,30) # bins=np.linspace(0.1,1.1,128))
        # plt.xlim([0.1,1.1])
        plt.grid(True)
        plt.xlabel("True values (normalized)")
        plt.ylabel("Count")
        plt.title(lbl_name+" true histogram (Ckpt "+ckpt_num+")")

        plt.subplot(232)
        plt.hist(predicted_labels,30) # bins=np.linspace(-1.0,1.0,128))
        # plt.xlim([-1,1])
        plt.grid(True)
        plt.xlabel("Predicted values (normalized)")
        plt.ylabel("Count")
        plt.title(lbl_name+" predictions histogram (Ckpt "+ckpt_num+")")

        plt.subplot(233)
        plt.hist(residuals,30)
        plt.xlabel("True - predicted (normalized)")
        plt.ylabel("Count")
        plt.title(lbl_name+" residuals histogram (Ckpt "+ckpt_num+")")

        buff = 0.1 # buffer for scatterplot axes limits; scaled to normalized data

        # Colored correlation scatterplot
        plt.subplot(234)
        plt.scatter(true_labels, predicted_labels, c=abs(residuals))
        plt.plot([0,1],[0,1],'-k')
        plt.xlim(min(true_labels)-buff, max(true_labels)+buff)
        plt.ylim(min(predicted_labels)-buff, max(predicted_labels)+buff)
        plt.xlabel("True labels")
        plt.ylabel("Predicted labels")
        plt.title(lbl_name+" true vs. predicted (Ckpt "+ckpt_num+")")

        # Residuals scatterplot
        plt.subplot(235)
        plt.scatter(true_labels, residuals)
        plt.plot([0,1],[0,0],'-b')
        plt.xlim(min(true_labels)-buff, max(true_labels)+buff)
        plt.ylim(min(residuals)-buff, max(residuals)+buff)
        plt.xlabel("True labels")
        plt.ylabel("Residuals")
        plt.title(lbl_name+" true vs. residuals (Ckpt "+ckpt_num+")")

        # Residuals scatterplot
        plt.subplot(236)
        plt.scatter(predicted_labels, residuals)
        plt.plot([0,1],[0,0],'-b')
        plt.xlim(min(true_labels)-buff, max(true_labels)+buff)
        plt.ylim(min(residuals)-buff, max(residuals)+buff)
        plt.xlabel("Predicted labels")
        plt.ylabel("Residuals")
        plt.title(lbl_name+" true vs. residuals (Ckpt "+ckpt_num+")")

    plt.show(block = FLAGS.show_eval_plots) # block = True means show plots

def evaluate(eval_files):
    _, meta_file = os.path.split(FLAGS.model_path)
    # Assume it looks like this : model.ckpt-9451.meta
    first_ckpt_num=meta_file.find("-")+1
    last_ckpt_num=len(meta_file)-5
    ckpt_num = meta_file[last_ckpt_num:first_ckpt_num]
    
    # Compile which labels were trained for
    lbls_trained_for = [FLAGS.train_p_t, FLAGS.train_p_z,
                     FLAGS.train_entry_x, FLAGS.train_entry_y, FLAGS.train_entry_z,
                     FLAGS.train_vert_x, FLAGS.train_vert_y, FLAGS.train_vert_z,
                     FLAGS.train_n_turns ]
    # for Saver
    ckpt_name = FLAGS.model_path[0:len(FLAGS.model_path)-5]
    
    with tf.Graph().as_default():
        # Extracting data
        pred_data = comet_dnn_input.read_tfrecord_to_dataset(
            eval_files,
            compression="GZIP",
            buffer_size=2e9,
            batch_size=FLAGS.batch_size,
            epochs=FLAGS.epochs,
            seed=FLAGS.random_seed)
        pred_iter = pred_data.make_one_shot_iterator()
        pred_images, true_labels = pred_iter.get_next()
        
        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        
        with tf.Session() as sess:
            print("Entering session...")
            # initialize variables
            sess.run(init_op)

            # Restore model, using Saver
            saver = tf.train.import_meta_graph(FLAGS.model_path)
            saver.restore(sess, ckpt_name)

            # Reloading predictions operation and corresponding placeholder variable
            graph = tf.get_default_graph()
            predictions = graph.get_tensor_by_name("predictions/predictions:0")
            batch_images = graph.get_tensor_by_name("input_images:0")
            
            # A BUNCH OF PRINT STATEMENTS FOR DEBUGGING MODEL RESTORE
            if FLAGS.debug_mode is True:
                tensor_to_print = "predictions/weights" # if want to print individual tensor from graph

                print("Printing tensor(s) in checkpoint file:")
                print_tensors_in_checkpoint_file(file_name=ckpt_name,
                                                 tensor_name=tensor_to_print,
                                                 all_tensors="",
                                                 all_tensor_names="")

                print(np.shape(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
                for tensor in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                    print(tensor.name)
                    print(graph.get_tensor_by_name(tensor_to_print+":0"))
                    print(graph.get_tensor_by_name(tensor_to_print+":0").eval())
                    
                    for op in graph.get_operations():
                        print(op.name)
                        print(op)
                print(graph.get_tensor_by_name("predictions/predictions:0"))

            # Get all true labels
            all_true_labels = true_labels.eval() # turn tensor into array
            # Move columns of predicted/desired true labels to front of array                                                              
            col_i = 0
            for lbl_i in range(len(lbls_trained_for)):
                if lbls_trained_for[lbl_i]:
                    all_true_labels[:,col_i] = all_true_labels[:,lbl_i]
                    col_i += 1
            true_labels = all_true_labels[:,:FLAGS.num_classes] # for easier printing, only take columns we bothered training for
            print("True labels:", true_labels)

            # Make feed_dict and run predictions operation
            print("Running predictions operation...")
            pred_feed = {batch_images: pred_images.eval()}
            predicted_labels = sess.run(predictions, feed_dict = pred_feed)
            print("Predictions:", predicted_labels)
 
            # Get some useful plots
            eval_plots(true_labels, predicted_labels, ckpt_num)


def main(argv=None):  # pylint: disable=unused-argument
    # Set the random seed
    FLAGS.random_seed = comet_dnn_input.set_global_seed(FLAGS.random_seed)
    # Dump the current settings to stdout
    comet_dnn.print_all_flags()
    # Read the input files and shuffle them
    # TODO read these from file list found in train_dir
    training_files, testing_files = \
        comet_dnn_input.train_test_split_filenames(FLAGS.input_list,
                                                   FLAGS.percent_train,
                                                   FLAGS.percent_test,
                                                   FLAGS.random_seed)
    # Evaluate the testing files by default
    eval_files = testing_files
    if not FLAGS.eval_test:
        eval_files = training_files
    # Evaluate the files
    evaluate(eval_files)
    
if __name__ == '__main__':
    tf.app.run()

# These scripts are based on the example of TensorFlow CIFAR-10
# They use the Apache lisence.  We will use the same.
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
"""

Summary of available functions:

# Compute input images and labels for training. If you would like to run
# evaluations, use inputs() instead.
inputs, labels = distorted_inputs()

# Compute inference on the model inputs to make a prediction.
predictions = inference(inputs)

# Compute the total loss of the prediction with respect to the labels.
loss = loss(predictions, labels)

# Create a graph to run one step of training with respect to the loss.
train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf
import comet_dnn_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_float('l_rate_init', 1e-7,
                          """Initial learning rate""")
tf.app.flags.DEFINE_float('l_rate_decay', 0.01,
                          """Decay of the learning rate by epoch""")
tf.app.flags.DEFINE_float('n_epochs_decay', 1,
                          """Number of epochs between each decay""")
tf.app.flags.DEFINE_float('move_avg_decay', 0.9999,
                          """Decay of the moving average""")
tf.app.flags.DEFINE_integer('batch_size', 256,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_integer('num_classes', 1,
                            """Number of classes you want to regress""")
tf.app.flags.DEFINE_integer('num_fc_nodes', 392,
                            """Number of fully connected nodes""")
tf.app.flags.DEFINE_float('weight_stddev', 5e-5,
                          """Standard deviation for weights""")
tf.app.flags.DEFINE_float('weight_decay', 5e-6,
                          """Standard deviation for weight decays""")
# TODO handle these inputs more dynamically as n_examples per file * n_files
#EXAMPLES_PER_EPOCH_FOR_TRAIN = 300000 # using 0.6 for now
EXAMPLES_PER_EPOCH_FOR_TRAIN = 55000 # using 0.8 for now
EXAMPLES_PER_EPOCH_FOR_EVAL = 100000  # using 0.2 for now

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def print_all_flags():
    print("Description ")
    # Print flags
    for flag, value in FLAGS.flag_values_dict().items():
        print(flag, "=", value)
    print("Examples per epoch for training:", EXAMPLES_PER_EPOCH_FOR_TRAIN)
    print("Examples per epoch for eval:", EXAMPLES_PER_EPOCH_FOR_EVAL)

def _activation_summary(a_tensor):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
    x: Tensor
    Returns:
    nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', a_tensor.op.name)
    tf.summary.histogram(tensor_name + '/activations', a_tensor)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(a_tensor))

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

    Returns:
    Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        return var

def _variable_with_weight_decay(name, shape, stddev, w_decay):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    w_decay: add L2Loss weight decay multiplied by this float. If None, weight
    decay is not added for this Variable.

    Returns:
    Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        #tf.truncated_normal_initializer(stddev=stddev, dtype=dtype, seed=FLAGS.random_seed))
        tf.contrib.layers.xavier_initializer( uniform=False,seed=FLAGS.random_seed))
    if w_decay is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), w_decay,
                                   name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def add_conv_lay(images, conv_shape, pool_shape, pool_strides,
                 layer_name="", kernel_name='weights', biases_name='biases',
                 stddev=5e-2, w_decay=None):
    """
    Reads a batch images, and returns norm and pool layer output.
    This function configures the shape of filter and stride for
    convolution layer and pool layer. What it does:
    1. Define kernel
    2. Do convolution with images and kernel
    3. Get biases on CPU
    4. Do bias_add ?
    5. Define activation function (TODO: make it optional)
    6. Add parameters to summary
    7. Do maximum pooling  (TODO: make it option)

    Parameters
    ----------
    images: tensor
        Batch of images tensor as an input
    conv_shape: tensor
        Shape of the kernal/filter for convolution layer
    pool_shape: tensor
        Shape of the filter for maximum pooling layer
    pool_strides: tensor
        Shape of the stride for maximum pooling layer
    layer_name: string
        Name of the layer, used to name the convolution, pooling, and
        normalization operations
    kernel_name: string, default weights
        Name of the kernel
    biases_name: string, default biases
        Name of the biases
    stddev: float, default 5e-2
        Standard deviation of a truncated Gaussian
    w_decay: float, default None
        Add L2Loss weight decay multiplied by w_decay.
        If None, weight is not added for this Variable.
    norm_par: list or sequence, default [4, 1.0, 0.001, 9.0, 0.75]
        Parameters for setting up the normalisation layer

    Returns
    ----------
    pool, norm :
        Output images of pooling layers
    """
    with tf.variable_scope("conv"+layer_name) as scope:
        # Initialize or get the kernel
        kernel = _variable_with_weight_decay(kernel_name,
                                             shape=conv_shape,
                                             stddev=stddev,
                                             w_decay=w_decay)

        # Perform the convolution
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        # Initialize or get the biases
        biases = _variable_on_cpu(biases_name, [conv_shape[3]],
                                  tf.constant_initializer(0.0))
        # Add the biases to the convolution
        pre_activation = tf.nn.bias_add(conv, biases)
        # Activate the layer
        activated = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(activated)
        # Max pool the results
        return tf.nn.max_pool(activated, ksize=pool_shape, strides=pool_strides,
                              padding='SAME', name="pool"+layer_name)

def inference(images):
    """
    Build the comet_dnn model. We have not finished testing
    out model yet (TODO, change here)

    Parameters
    ----------
    images: tensor
        Images returned from distorted_inputs() or inputs().

    Returns
    -------
    predictions: tensor
        Predictions after filtering by the model. This should be
        a tensor of [batch_size, num_classes].
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training
    # runs. If we only ran this model on a single GPU, we could simplify this
    # function by replacing all instances of tf.get_variable() with
    # tf.Variable().

    # First convolution layer, conv1
    conv_shape = [1, 2, 2, 16]
    pool_shape = [1, 1, 2, 1]
    pool_strides = [1, 1, 2, 1]
    conv1 = add_conv_lay(images, conv_shape, pool_shape, pool_strides,
                         layer_name="1")
    # Normalize the output
    norm = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75,
                     name="norm1")
    # Second convolution layer
    conv_shape = [1, 2, 16, 16]
    conv2 = add_conv_lay(norm, conv_shape, pool_shape, pool_strides,
                         layer_name="2")

    # fully_connected1
    with tf.variable_scope('fully_connected1') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(conv2, [images.get_shape().as_list()[0], -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, FLAGS.num_fc_nodes],
                                              stddev=FLAGS.weight_stddev, w_decay=FLAGS.weight_decay)
        biases = _variable_on_cpu('biases', [FLAGS.num_fc_nodes], tf.constant_initializer(0.1))
        fully_connected1 = tf.nn.relu(tf.matmul(reshape, weights) + biases,
                                      name=scope.name)
        _activation_summary(fully_connected1)

    # fully_connected2
    with tf.variable_scope('fully_connected2') as scope:
        weights = _variable_with_weight_decay('weights', shape=[FLAGS.num_fc_nodes, FLAGS.num_fc_nodes/2],
                                              stddev=FLAGS.weight_stddev, w_decay=FLAGS.weight_decay)
        biases = _variable_on_cpu('biases', [FLAGS.num_fc_nodes/2], tf.constant_initializer(0.1))
        fully_connected2 = \
                tf.nn.relu(tf.matmul(fully_connected1, weights) + biases,
                           name=scope.name)
        _activation_summary(fully_connected2)

    # Get the final predictions
    with tf.variable_scope('predictions') as scope:
        weights = _variable_with_weight_decay('weights',
                                              [FLAGS.num_fc_nodes/2, FLAGS.num_classes],
                                              stddev=1/FLAGS.num_fc_nodes/2.0, w_decay=None)
        biases = _variable_on_cpu('biases', [FLAGS.num_classes],
                                  tf.constant_initializer(0.0))
        predictions = tf.add(tf.matmul(fully_connected2, weights), biases,
                             name=scope.name)
        _activation_summary(predictions)

    return predictions

def loss(predictions, labels):
    """
    Calculate the losses based on the predictions
    Parameters
    ----------
    predictions: tensor
        Predictions from inference of comet_dnn.
        The size should be [batch_size, num_classes]
    labels: tensor
        Labels for the image. It must be the same size
        as predictions
    """
    # Calculate residual of labels and predictions and the mean across the
    # batch for each label, i.e. mean_resid has shape (FLAGS.num_classes)
    residuals = tf.subtract(labels[:, :FLAGS.num_classes], predictions)
    mean_residuals = tf.reduce_mean(residuals, axis=0)
    # Get the square and the mean for each label, as above
    sqr_residuals = tf.square(residuals)
    mean_sqr_residuals = tf.reduce_mean(sqr_residuals, axis=0)
    # Get the RMS value
    root_mean_square = tf.sqrt(mean_sqr_residuals)
    # Get the loss as the mean
    total_loss = tf.reduce_mean(mean_sqr_residuals)
    # Then add to collection
    tf.add_to_collection('losses', total_loss)

    # Save the residuals and root mean square for each function
    for i in range(FLAGS.num_classes):
        # Fill histogram of residual
        name = comet_dnn_input.LABEL_NAMES[i]
        tf.summary.histogram("pre", predictions[:, i] , family=name)
        tf.summary.histogram("lab", labels[:, i]      , family=name)
        tf.summary.histogram("res", residuals[:, i]   , family=name)
        # Fill scalar summaries of the mean residuals and the RMS
        tf.summary.scalar("mean", mean_residuals[i]   , family=name)
        tf.summary.scalar("rms" , root_mean_square[i] , family=name)

    # Return the total loss
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
    """Add summaries for losses in comet_dnn model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
    total_loss: Total loss from loss().
    Returns:
    loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do
    # the same for the averaged version of the losses.
    for a_l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
        tf.summary.scalar(a_l.op.name + ' (raw)', a_l)
        tf.summary.scalar(a_l.op.name, loss_averages.average(a_l))
    return loss_averages_op


def train(total_loss, global_step):
    """Train comet_dnn model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
    processed.
    Returns:
    train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * FLAGS.n_epochs_decay)

    # Decay the learning rate exponentially based on the number of steps.
    learn_r = tf.train.exponential_decay(FLAGS.l_rate_init,
                                         global_step,
                                         decay_steps,
                                         FLAGS.l_rate_decay,
                                         staircase=True)
    tf.summary.scalar('learning_rate', learn_r)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(learn_r)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    var_averages = tf.train.ExponentialMovingAverage(FLAGS.move_avg_decay,
                                                     global_step)
    with tf.control_dependencies([apply_gradient_op]):
        vars_averages_op = var_averages.apply(tf.trainable_variables())

    return vars_averages_op

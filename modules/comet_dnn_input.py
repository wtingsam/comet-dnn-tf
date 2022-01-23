# These scripts are based on the example of TensorFlow CIFAR-10
# They use the Apache lisence.  We will use the same.
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================

"""Routine for decoding the comet_dnn binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np

import tensorflow as tf

# Define our image shape as a constant
IMAGE_SHAPE = [18, 300, 2]
LABEL_SHAPE = [9]
FLAGS = tf.app.flags.FLAGS

# Input flags
tf.app.flags.DEFINE_string('input_list', None,
                           """A list contains all paths of .tfrecords""")
tf.app.flags.DEFINE_float('percent_train', 0.6,
                          """Percentage of sample that goes into the training
                             set""")
tf.app.flags.DEFINE_float('percent_test', 0.2,
                          """Percentage of sample that goes into the test
                             set""")
tf.app.flags.DEFINE_integer('epochs', 10,
                            """Number of training epochs to iterate through""")
tf.app.flags.DEFINE_integer('input_buffer_size', 0,
                            """Input buffer size in bytes""")
tf.app.flags.DEFINE_integer('max_output_images', 10,
                            """Number of images you want to save""")
tf.app.flags.DEFINE_integer('random_seed', None,
                            """Set the graph level random seed, including
                               randomization of the train/test splitting and
                               input shuffling.  Defaults to random number from
                               [0, max uint64)""")

# For choosing  arbitrary set of labels to train on (T/F decided using integer)
# If all are 0, labels are trained in order up to num_classes
tf.app.flags.DEFINE_integer("train_p_t", 0,
                            """Whether to train on p_t""")
tf.app.flags.DEFINE_integer("train_p_z", 0,
                            """Whether to train on p_z""")
tf.app.flags.DEFINE_integer("train_entry_x", 0,
                            """Whether to train on entry_x""")
tf.app.flags.DEFINE_integer("train_entry_y", 0,
                            """Whether to train on entry_y""")
tf.app.flags.DEFINE_integer("train_entry_z", 0,
                            """Whether to train on entry_z""")
tf.app.flags.DEFINE_integer("train_vert_x", 0,
                            """Whether to train on vert_x""")
tf.app.flags.DEFINE_integer("train_vert_y", 0,
                            """Whether to train on vert_y""")
tf.app.flags.DEFINE_integer("train_vert_z", 0,
                            """Whether to train on vert_z""")
tf.app.flags.DEFINE_integer("train_n_turns", 0,
                            """Whether to train on n_turns""")

# Global constants describing the comet_dnn data set.
EXAMPLES_PER_FILE = 5000

LABEL_NAMES = ["p_t", "p_z",
               "entry_x", "entry_y", "entry_z",
               "vert_x", "vert_y", "vert_z",
               "n_turns"]

LABEL_TITLES = ['Transverse_Momentum',
                'Longituadinal_Momentum',
                'Entry_of_X',
                'Entry_of_Y',
                'Entry_of_Z',
                'Vertex_of_X',
                'Vertex_of_Y',
                'Vertex_of_Z',
                'Number_of_turns']

LABEL_NORMALIZE = [110., 80.,
                   60., 60., 70.,
                   11., 11., 45.,
                   5.]
FEATURE_NORMALIZE = [10., 1000.]

def write_list_to_file(a_list, filename):
    """
    Write a list to a file, one line per entry

    Parameters
    ----------
    a_list: list
        List to be written to file
    file_name: string
        Path to the output file
    """
    with open(filename, 'w') as out_file:
        out_file.write("\n".join(a_list))

def write_array_to_tfrecord(array, labels, filename, options=None):
    # Open TFRecords file, ensure we use gzip compression
    writer = tf.python_io.TFRecordWriter(filename, options=options)

    # Write all the images to a file
    for lbl, img in zip(labels, array):
        # Create the feature dictionary and enter the image
        image_as_bytes = tf.train.BytesList(
            value=[tf.compat.as_bytes(img.tostring())])
        feature = {'image':  tf.train.Feature(bytes_list=image_as_bytes)}
        # Create anentry for each label
        for a_lab, name_lab in zip(lbl, LABEL_NAMES):
            label_as_float = tf.train.FloatList(value=[a_lab])
            feature[name_lab] = tf.train.Feature(float_list=label_as_float)
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    # Close the writer and flush the buffer
    writer.close()
    sys.stdout.flush()

def set_global_seed(seed=None):
    """
    Sets the global to the argument.  If the argument is none, it uses numpy to
    generate a new random seed and sets the global seed to this. The seed is set
    for both numpy and tensorflow.

    Parameters
    ----------
    seed : number
        The seed to set globally.  If it is None, one is generated.

    Returns
    -------
    global_seed: number
        The number that the random seed was set to.
    """
    # Check if we need to generate a seed
    set_seed = seed
    if seed is None:
        set_seed = np.random.randint(np.iinfo(np.uint32).max, dtype='uint32')
    # Set this seed to all of the packages that need it
    tf.set_random_seed(set_seed)
    np.random.seed(set_seed)
    # Return whatever number was used
    return set_seed

def train_test_split_filenames(filelist_path,
                               percent_train,
                               percent_test,
                               seed):
    """
    Read a list of file names from a text file.  Randomly shuffle and split the
    list of file names into training and testing subsamples.  Of the total
    number of files, int(n_total_files * percent_train) files will be returned
    in the training set list, while int(n_total_files * percent_test) will be
    returned in the testing set list.

    Note that this must be seeded.

    Parameters
    ----------
    filelist_path : string
        Path to text file of file names, one per line
    percent_train : float
        Proportion of total files to return in the training set
    percent_test : float
        Propotion of total files to return in the testing set
    seed: int, float
        Seed for the randomization of the file list

    Returns
    -------
    train_list, test_list : list, list
        Two lists of strings, with each containing paths to the input files
    """
    # Get all the file names
    all_filenames = open(filelist_path, 'r')
    list_filenames = all_filenames.read().splitlines()
    all_filenames.close()
    # Ensure we are not asking for too many files
    assert int((percent_train + percent_test)*100) <= 100,\
        "Ensure that percent_train + percent_test <= 1.00"+\
        " %.02f : percent_train" % percent_train +\
        " %.02f : percent_test" % percent_test
    # Get the current random state
    old_state = np.random.get_state()
    # Set the seed to the needed value
    np.random.seed(seed)
    # Randomize the file list
    list_filenames = np.array(list_filenames)
    np.random.shuffle(list_filenames)
    # Reset the random state
    np.random.set_state(old_state)
    # Grab the first percent_train percentage as the set of training sample
    total_n_files = list_filenames.size
    last_train_file = int(total_n_files * percent_train)
    training_files = list_filenames[0:last_train_file]
    # Grabe the next percentage as testing files
    last_test_file = last_train_file + int(total_n_files * percent_test)
    testing_files = list_filenames[last_train_file:last_test_file]
    # Debug messages
    if(FLAGS.debug_mode):
        print("Number of training files :",
              last_train_file)
        print(training_files)
        print("Number of testing files  :",
              last_test_file - last_train_file)
        print(testing_files)

    # Ensure we have not asked for too few files
    assert training_files.size > 0,\
        "Asked for less than one training file!"
    assert testing_files.size > 0,\
        "Asked for less than one testing file!"
    # Return the file lists
    return list(training_files), list(testing_files)

def parse_record_into_tensors(record):
    # Compile which labels were trained for
    lbls_to_train = [FLAGS.train_p_t, FLAGS.train_p_z,
		     FLAGS.train_entry_x, FLAGS.train_entry_y, FLAGS.train_entry_z,
                     FLAGS.train_vert_x, FLAGS.train_vert_y, FLAGS.train_vert_z,
                     FLAGS.train_n_turns ]
    # Re-order label names for correct summary saving names
    i = 0
    for j in range(len(lbls_to_train)):
        if lbls_to_train[j]:
            LABEL_NAMES[i] = LABEL_NAMES[j]
            i += 1
    print(LABEL_NAMES[:FLAGS.num_classes])
    # Decode the features in the dataset
    features = {'image': tf.FixedLenFeature([], tf.string)}
    for name in LABEL_NAMES:
        features[name] = tf.FixedLenFeature([], tf.float32)
    # Parse the example
    parsed_features = tf.parse_single_example(record, features)
    # Decode and reshape the image
    image = tf.decode_raw(parsed_features['image'], tf.float32)
    image = tf.reshape(image, [18, 300, 2])
    
    image_norm = tf.constant(FEATURE_NORMALIZE,
                             dtype=tf.float32,
                             name="image_norms")
    image_normed = tf.div(image,image_norm)
    # Stack the labels
    all_labels = tf.stack([tf.cast(parsed_features[name], tf.float32)
                           for name in LABEL_NAMES])
    label_norm = tf.constant(LABEL_NORMALIZE,
                             dtype=tf.float32,
                             name="label_norms")
    all_labels_normed = tf.div(all_labels, label_norm)
    tf.summary.histogram("charge"   ,image_normed[:,:,0], family="image")
    tf.summary.histogram("drifttime",image_normed[:,:,1], family="image")
    return image_normed, all_labels_normed

def read_tfrecord_to_dataset(filenames,
                             compression="GZIP",
                             buffer_size=0,
                             batch_size=256,
                             epochs=1,
                             seed=None):
    """
    Reads a list of TFRecords into a TFRecordDataset, and returns the Dataset
    iterators that yield (images, labels) pairs.  This function configures the
    batch-size, number of epochs, and buffer-size for reading in events.

    Parameters
    ----------
    filenames: list or sequence
        List of input file names
    compression: string, default "GZIP"
        Compression options for the TFRecord.  Can be "", "ZLIB", "GZIP"
    buffer_size: int, default is 0, no buffer
        Size in bytes for reading in TFRecord files
    batch_size: int, default is 256
        Size of the batch to use
    epochs : int, optional, default 1
        Number of epochs for the data (number of times each input file in the
        file list can be read

    Returns
    -------
    images, labels: Tensor, Tensor
        Images of each event are [batch_size, 18, 300]
        Labels are [batch_size, 9]
    """
    # Open a variable scope for the input data
    with tf.variable_scope('input_data') as _:
        # Get a tensor of all files and suffle it
        in_files = tf.convert_to_tensor(filenames)
        in_files = tf.data.Dataset.from_tensor_slices(in_files)
        in_files = in_files.shuffle(len(filenames), seed=seed)
        # Initialize the tf.dataset from the input filenames of the tfrecords
        tf_dataset = tf.data.TFRecordDataset(in_files,
                                             compression_type=compression,
                                             buffer_size=int(buffer_size))
        # Parse the tfrecord into the tesnor values
        tf_dataset = tf_dataset.map(parse_record_into_tensors)
        # Shuffle the dataset
        tf_dataset = tf_dataset.shuffle(buffer_size=EXAMPLES_PER_FILE, seed=seed)
        # Repeat the dataset for a given number of epochs
        tf_dataset = tf_dataset.repeat(epochs)
        # Set the batch size and ignore the last few elements
        tf_batch = tf.contrib.data.batch_and_drop_remainder(batch_size)
        tf_dataset = tf_dataset.apply(tf_batch)
        # Return the dataset
        return tf_dataset

"""
  preprocess_xview_dataset_test_parse.py
  SEIS764 - Artificial Intelligence
  John Wagener, Kwame Sefah, Gavin Pedersen, Vong Vang, Zach Aaberg

  Verify that the files written are readable.
"""
# Revised for our dataset from:
# https://www.tensorflow.org/neural_structured_learning/tutorials/graph_keras_mlp_cora
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import neural_structured_learning as nsl

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# Resets notebook state
tf.keras.backend.clear_session()

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

### Experiment dataset
TEST_DATA_PATH = '../../data/filtered_images_for_training_tf/test.tfr'
TRAIN_DATA_PATH = '../../data/filtered_images_for_training_tf/train.tfr'

### Constants used to identify neighbor features in the input.
NBR_FEATURE_PREFIX = 'NL_nbr_'
NBR_WEIGHT_SUFFIX = '_weight'


class HParams(object):
    def __init__(self):
        # JDW set image size
        # self.input_shape = [255, 255, 1]
        self.input_shape = [1024, 1024, 3]
        # JDW pass in max lenght used for default of 0
        self.max_seq_length = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]
        # TODO JDW Set number of neighbors
        self.num_neighbors = 8
        # **************************************
        self.num_classes = 4
        # JDW Crashes with 64
        # self.conv_filters = [32, 64, 64]
        self.conv_filters = [32, 32, 32]
        self.kernel_size = (3, 3)
        self.pool_size = (2, 2)
        self.num_fc_units = [64]
        # JDW Set back to 32
        self.batch_size = 4
        self.epochs = 5
        self.adv_multiplier = 0.2
        self.adv_step_size = 0.2
        self.adv_grad_norm = 'infinity'


def parse_example(example_proto):
    """Extracts relevant fields from the `example_proto`.

    Args:
      example_proto: An instance of `tf.train.Example`.

    Returns:
      A pair whose first value is a dictionary containing relevant features
      and whose second value contains the ground truth labels.
    """
    # The 'image' feature is a image representation of the
    # original raw image. A default value is required for examples that don't
    # have the feature.
    feature_spec = {
        # 'image_x':
        #    tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        # 'image_y':
        #    tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        # 'image_filename':
        #    tf.io.VarLenFeature(tf.string),
        'label':
            tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        'image':
            tf.io.FixedLenFeature([], tf.string),
    }
    # We also extract corresponding neighbor features in a similar manner to
    # the features above.
    print("HERE******************")
    for i in range(HPARAMS.num_neighbors):
        nbr_feature_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX, i, 'image')
        nbr_weight_key = '{}{}{}'.format(NBR_FEATURE_PREFIX, i, NBR_WEIGHT_SUFFIX)
        print("nbr_feature_key=" + nbr_feature_key)
        feature_spec[nbr_feature_key] = tf.io.FixedLenFeature(
            [],
            tf.string,
            default_value=tf.constant(
                0, dtype=tf.string, shape=[HPARAMS.max_seq_length]))
        #feature_spec[nbr_feature_key] = tf.io.FixedLenFeature(
        #    [HPARAMS.max_seq_length],
        #    tf.string,
        #    default_value=tf.constant(
        #        0, dtype=tf.string, shape=[HPARAMS.max_seq_length]))
        #feature_spec[nbr_feature_key] = {
            # 'image_x':
            #    tf.io.FixedLenFeature([], tf.int64, default_value=-1),
            # 'image_y':
            #    tf.io.FixedLenFeature([], tf.int64, default_value=-1),
            # 'image_filename':
            #    tf.io.VarLenFeature(tf.string),
        #    nbr_feature_key + '_label':
        #        tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        #    nbr_feature_key:
        #        tf.io.FixedLenFeature([], tf.string),
        #}

        # We assign a default value of 0.0 for the neighbor weight so that
        # graph regularization is done on samples based on their exact number
        # of neighbors. In other words, non-existent neighbors are discounted.
        feature_spec[nbr_weight_key] = tf.io.FixedLenFeature([1], tf.float32, default_value=tf.constant([0.0]))

    print(feature_spec.keys())
    features = tf.io.parse_single_example(example_proto, feature_spec)

    labels = features.pop('label')
    return features, labels


def make_dataset(file_path, training=False):
    """Creates a `tf.data.TFRecordDataset`.

    Args:
      file_path: Name of the file in the `.tfrecord` format containing
        `tf.train.Example` objects.
      training: Boolean indicating if we are in training mode.

    Returns:
      An instance of `tf.data.TFRecordDataset` containing the `tf.train.Example`
      objects.
    """
    dataset = tf.data.TFRecordDataset([file_path])
    if training:
        dataset = dataset.shuffle(10000)
    dataset = dataset.map(parse_example)
    dataset = dataset.batch(HPARAMS.batch_size)
    return dataset


####################
HPARAMS = HParams()
train_dataset = make_dataset(TRAIN_DATA_PATH, training=True)
#test_dataset = make_dataset(TEST_DATA_PATH)

abc = train_dataset.take(1)
print(abc)
for feature_batch, label_batch in abc:
    print('Feature list:', list(feature_batch.keys()))
    print('Batch of inputs:', feature_batch['image_filename'])
    # print('Batch of inputs:', feature_batch['image'])
    nbr_feature_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX, 0, 'image')
    nbr_weight_key = '{}{}{}'.format(NBR_FEATURE_PREFIX, 0, NBR_WEIGHT_SUFFIX)
    print('Batch of neighbor inputs:', feature_batch[nbr_feature_key])
    print('Batch of neighbor weights:',
          tf.reshape(feature_batch[nbr_weight_key], [-1]))
    print('Batch of labels:', label_batch)

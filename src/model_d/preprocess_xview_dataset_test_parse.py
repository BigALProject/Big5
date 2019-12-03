"""
  preprocess_xview_dataset_test_parse.py
  SEIS764 - Artificial Intelligence
  John Wagener, Kwame Sefah, Gavin Pedersen, Vong Vang, Zach Aaberg

  Verify that the files written are readable.

  Revised for our dataset from:
  https://www.tensorflow.org/neural_structured_learning/tutorials/graph_keras_mlp_cora
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

class HParams(object):
    def __init__(self):
        # set image size
        self.input_shape = [1024, 1024, 3]

        # compute length
        self.max_seq_length = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]

        # set number of neighbors
        # Note some items had only 7 neighbors, not sure why, so lets use 7.
        self.num_neighbors = 8

        # no-damage, minor-damage, major-damage, destroyed
        self.num_classes = 4
        # Note: My machine crashes with 64
        # self.conv_filters = [32, 64, 64]
        self.conv_filters = [32, 32, 32]
        self.kernel_size = (3, 3)
        self.pool_size = (2, 2)
        # Number of fully connected units
        self.num_fc_units = [64]
        # Training info
        self.batch_size = 1
        self.epochs = 5
        # JDW: WHAT ARE THESE?
        self.adv_multiplier = 0.2
        self.adv_step_size = 0.2
        self.adv_grad_norm = 'infinity'


def parse_example(example_proto):
    global feature_spec
    """
    Extracts relevant fields from the `example_proto`.

    Args:
      example_proto: An instance of `tf.train.Example`.
      feature_spec: note that train is different than test because test does not include neighbors

    Returns:
      A pair whose first value is a dictionary containing relevant features
      and whose second value contains the ground truth labels.
    """
    features = tf.io.parse_single_example(example_proto, feature_spec)
    labels = features.pop('label')
    return features, labels

def make_dataset(file_path, training=False):
    global feature_spec
    """Creates a `tf.data.TFRecordDataset`.

    Args:
      file_path: Name of the file in the `.tfrecord` format containing
        `tf.train.Example` objects.
      training: Boolean indicating if we are in training mode.

    Returns:
      An instance of `tf.data.TFRecordDataset` containing the `tf.train.Example`
      objects.
    """

    # The feature spec node contains features for the node.
    feature_spec_node = {
        'label':
            tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        'image_filename':
            tf.io.FixedLenFeature([], tf.string, default_value="NO_IMAGE"),
        # Decided not to persist the images + metadata in the tensorflow file as we'll read it when processing.
        #'image':
        #    tf.io.FixedLenFeature([], tf.string),
        # 'image_x':
        #    tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        # 'image_y':
        #    tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        #'image_non_zero':
        #    tf.io.FixedLenFeature([], tf.int64),
    }

    # Master feature spec
    feature_spec = {}
    feature_spec.update(feature_spec_node)

    if training:
        # Add number of neighbors feature
        feature_spec['NL_num_nbrs'] = tf.io.FixedLenFeature([], tf.int64)

        # We also extract corresponding neighbor features in a similar manner to
        # the features above.  Only for training.
        for i in range(HPARAMS.num_neighbors):
            nbr_prefix = '{}{}_'.format(NBR_FEATURE_PREFIX, i)
            feature_spec_neighbor = {
                nbr_prefix + 'label':
                    tf.io.FixedLenFeature([], tf.int64, default_value=-1),
                nbr_prefix + 'image_filename':
                    tf.io.FixedLenFeature([], tf.string, default_value="NO_IMAGE"),
                # Weight is the distance.  Not sure of this effect on the training...
                nbr_prefix + 'weight':
                    tf.io.FixedLenFeature([], tf.float32, default_value=tf.constant(1.0)),
                #nbr_prefix + 'image':
                #    tf.io.FixedLenFeature([], tf.string),
                #nbr_prefix + 'image_non_zero':
                #    tf.io.FixedLenFeature([], tf.int64),
            }

            feature_spec.update(feature_spec_neighbor)

    # load the data form the file.
    dataset = tf.data.TFRecordDataset([file_path])
    if training:
        dataset = dataset.shuffle(10000)
    dataset = dataset.map(parse_example)
    dataset = dataset.batch(HPARAMS.batch_size)
    return dataset

# Global (cant seem to pas this in correctly.
feature_spec = {}

if __name__ == '__main__':
    # Files to read.
    TRAIN_DATA_PATH = '../../data/filtered_images_for_training_tf/train.tfr'
    TEST_DATA_PATH = '../../data/filtered_images_for_training_tf/test.tfr'

    # Constants used to identify neighbor features in the input.
    NBR_FEATURE_PREFIX = 'NL_nbr_'
    NBR_WEIGHT_SUFFIX = '_weight'

    # Create hyper parameters
    HPARAMS = HParams()

    print("Creating training dataset...")
    train_dataset = make_dataset(TRAIN_DATA_PATH, training=True)
    print("Creating test dataset...")
    test_dataset = make_dataset(TEST_DATA_PATH)

    print("---- Example dataset ---")
    batch_counter = 0
    # Use this to dump 1 item.
    # for feature_batch, label_batch in train_dataset.take(1):
    # Use this to dump everything.
    for feature_batch, label_batch in train_dataset:
        batch_counter = batch_counter + 1
        print("-- BATCH counter=" + str(batch_counter))
        print('  Feature list                  :', list(feature_batch.keys()))
        print('  Batch of image_filenames      :', feature_batch['image_filename'])
        # Only print this if we're reading images and generating metadata.
        # print('  Batch of image_no_zero counts :', feature_batch['image_non_zero'])
        print('  Batch of NL_num_nbrs          :', feature_batch['NL_num_nbrs'])
        for i in range(HPARAMS.num_neighbors):
            print("  ---- neighbor i=" + str(i))
            prefix = '{}{}_'.format(NBR_FEATURE_PREFIX, i)
            print('    Neighbor inputs:', feature_batch[prefix + "image_filename"])
            print('    Neighbor weights:',
                  tf.reshape(feature_batch[prefix + "weight"], [-1]))
            print('    Neighbor Labels:', label_batch)

    print("---- All training nodes ---")
    for feature_batch, label_batch in train_dataset:
        print('  Batch of image_filenames      :', feature_batch['image_filename'])
    print("---- All test nodes ---")
    for test_feature_batch, test_label_batch in test_dataset:
        print('  Batch of image_filenames      :', test_feature_batch['image_filename'])

    #print("----- Summary -----")
    #print("Total nodes=" + str(len(feature_batch['image_filename']) + len(test_feature_batch['image_filename'])))
    print("----- Summary -----")
    print("batch_counter=" + str(batch_counter))
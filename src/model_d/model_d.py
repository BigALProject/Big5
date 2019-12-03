"""
  model_d.py
  SEIS764 - Artificial Intelligence
  John Wagener, Kwame Sefah, Gavin Pedersen, Vong Vang, Zach Aaberg

  Uses G

  Next: preprocess_xview_dataset_test_parse.py

  Revised for our dataset from:
https://www.tensorflow.org/neural_structured_learning/tutorials/graph_keras_mlp_cora
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import cv2

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

import neural_structured_learning as nsl
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Hack to turn off GPU, for now.
# from keras import Input
import numpy as np


class HParams(object):
    def __init__(self):
        # set image size
        self.input_shape = [IMG_HEIGHT, IMG_WIDTH, DEPTH]

        # compute length
        self.max_seq_length = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]

        ### neural graph learning parameters
        # Note some items had only 7 neighbors, not sure why, so lets use 7.
        self.num_neighbors = 7
        self.distance_type = nsl.configs.DistanceType.L2
        self.graph_regularization_multiplier = 0.1

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
        # TODO JDW FID ME
        self.epochs = 1
        self.dropout_rate = .5
        # JDW: WHAT ARE THESE?
        self.eval_steps = None  # All instances in the test set are evaluated.
        self.adv_multiplier = 0.2
        self.adv_step_size = 0.2
        self.adv_grad_norm = 'infinity'


def transform_images(row, size):
    x_train = tf.image.resize(row['image'], (size, size))
    x_train = x_train / 255
    return x_train, row['coarse_label'], row['label']


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


#def _parse_(example_proto):
#    global feature_spec
#    """
#    Extracts relevant fields from the `example_proto`.
#
#    Args:
#      example_proto: An instance of `tf.train.Example`.
#      feature_spec: note that train is different than test because test does not include neighbors
#
#    Returns:
#      A pair whose first value is a dictionary containing relevant features
#      and whose second value contains the ground truth labels.
#    """
#    features = tf.io.parse_single_example(example_proto, feature_spec)
#    image = tf.decode_raw(features['image'], tf.int64)  # remember to parse in int64. float will raise error
#    label = tf.cast(features['label'], tf.int32)
#
#    # add in additonal images here...
#    return dict({'image': image}), label

#def tfrecord_train_input_fn(file_path, batch_size=32):
#    tfrecord_dataset = tf.data.TFRecordDataset([file_path])
#    tfrecord_dataset = tfrecord_dataset.map(lambda x: _parse_(x)).shuffle(True).batch(batch_size)
#   tfrecord_iterator = tfrecord_dataset.make_one_shot_iterator()
#    return tfrecord_iterator.get_next()

#def make_dataset2(filename_queue, training=False):
#    reader = tf.TFRecordReader()
#
#    _, serialized_example = reader.read(filename_queue[0])
#
#    features = tf.parse_single_example(
#        serialized_example,
#        # Defaults are not specified since both keys are required.
#        # The feature spec node contains features for the node.
#        features={
#            'image_x':
#                tf.io.FixedLenFeature([], tf.int64, default_value=-1),
#            'image_y':
#                tf.io.FixedLenFeature([], tf.int64, default_value=-1),
#            'label':
#                tf.io.FixedLenFeature([], tf.int64, default_value=-1),
#            'image':
#                tf.io.FixedLenFeature([], tf.string),
#            'image_filename':
#                tf.io.FixedLenFeature([], tf.string),
#            # 'image_non_zero':
#            #    tf.io.FixedLenFeature([], tf.int64),
#         })
#
#    # Convert from a scalar string tensor (whose single string has
#    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
#    # [mnist.IMAGE_PIXELS].
#    image = tf.decode_raw(features['image'], tf.uint8)
#    height = tf.cast(features['image_y'], tf.int64)
#    width = tf.cast(features['image_x'], tf.int64)
#
#    image_shape = tf.pack([height, width, 3])
#    image = tf.reshape(image, image_shape)
#    image_size_const = tf.constant((IMG_HEIGHT, IMG_WIDTH, DEPTH), dtype=tf.int64)
#
#    # Random transformations can be put here: right before you crop images
#    # to predefined size. To get more information look at the stackoverflow
#    # question linked above.
#
#    resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
#                                                           target_height=IMG_HEIGHT,
#                                                           target_width=IMG_WIDTH)
#
#    images = tf.train.shuffle_batch([resized_image],
#                                    batch_size=2,
#                                    capacity=30,
#                                    num_threads=2,
#                                    min_after_dequeue=10)
#
#    labels = features.pop('label')
#    features['image'] = images
#    return features, labels


def augment(x, y):
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    #x['image'] = tf.image.resize_with_crop_or_pad(x['image'], HEIGHT + 8, WIDTH + 8)
    #print(x['image'])
    #x['image'] = tf.image.random_crop(x['image'], [IMG_HEIGHT, IMG_WIDTH, DEPTH])
    #print(type(x['image']))
    #x['image'] = tf.reshape(x['image'], [-1])
    #print(x['image'])
    #x['image'] = tf.reshape(x['image'], [IMG_WIDTH, IMG_WIDTH, DEPTH])
    print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
    print(x)
    print(x['image'])
    print(x['image_filename'])
    print("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")
    #x['image'] = cv2.imread(
    #    x['image_filename'],
    #    1)
    #image = tf.io.read_file(x['image_filename'])
    print("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
    #x['image'] = image
    # TODO THIS WORKS.... NEED TO READ IN IMAGE LATE
    x['image'] = cv2.imread(
        '../../data/filtered_images_for_training/no-damage\\2d6341c5-299b-4834-aca9-bfead35d4c83_no-damage_hurricane-harvey_00000008_post_disaster.png',
        1)
    print("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
    return x, y


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
        # 'image_x':
        #    tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        # 'image_y':
        #    tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        'label':
            tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        'image':
            tf.io.FixedLenFeature([], tf.string),
        'image_filename':
            tf.io.FixedLenFeature([], tf.string),
        # 'image_non_zero':
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
                #nbr_prefix + 'image':
                #    tf.io.FixedLenFeature([], tf.string),
                nbr_prefix + 'image_filename':
                    tf.io.FixedLenFeature([], tf.string),
                # nbr_prefix + 'image_non_zero':
                #    tf.io.FixedLenFeature([], tf.int64),
                # nbr_prefix + 'weight':
                #    tf.io.FixedLenFeature([], tf.float32, default_value=tf.constant(1.0)),
            }
            # Need to load neighbor images here.
            feature_spec.update(feature_spec_neighbor)

    # load the data form the file.
    dataset = tf.data.TFRecordDataset([file_path])
    # if training:
    #     dataset = dataset.shuffle(10000)
    dataset = dataset.map(parse_example)
    dataset = dataset.map(augment)
    dataset = dataset.batch(HPARAMS.batch_size)

    # import matplotlib.pyplot as plt
    #
    # for image_features in dataset:
    #    image_raw = image_features[0]['image'].numpy()
    #    print(type(image_features[0]['image']))
    #    plt.imshow(lambda x: image_features[0]['image'].cast(x, tf.float32))
    #    plt.show()
    return dataset


def print_metrics(model_desc, eval_metrics):
    """Prints evaluation metrics.

    Args:
      model_desc: A description of the model.
      eval_metrics: A dictionary mapping metric names to corresponding values. It
        must contain the loss and accuracy metrics.
    """
    print('\n')
    print('Eval accuracy for ', model_desc, ': ', eval_metrics['accuracy'])
    print('Eval loss for ', model_desc, ': ', eval_metrics['loss'])
    if 'graph_loss' in eval_metrics:
        print('Eval graph loss for ', model_desc, ': ', eval_metrics['graph_loss'])


def simple_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                               name='image'),
        # tf.keras.layers.MaxPooling2D(pool_size=HPARAMS.pool_size, strides=2, padding='valid', data_format='channels_last'),
        # tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        # tf.keras.layers.MaxPooling2D(pool_size=HPARAMS.pool_size, strides=2, padding='valid', data_format='channels_last'),
        # tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        # tf.keras.layers.MaxPooling2D(pool_size=HPARAMS.pool_size, strides=2, padding='valid', data_format='channels_last'),
        # tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='softmax')
    ])
    return model


def create_vgg():
    # Load the VGG16 model
    # resize default image size to 128x128 to speed up model
    vgg_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False,
                                            input_tensor=tf.keras.Input(shape=(1024, 1024, 3)), )
    # Add FC layer and output layer
    x = vgg_model.output
    x = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(4, activation='softmax', kernel_initializer='random_normal', name='predict_classes')(
        x)  # connect all layers I've created

    my_model = tf.keras.Model(inputs=vgg_model.input, outputs=x)
    # Freeze VGG16 base layers so weights are not be updated during backpropagation
    for layer in vgg_model.layers:
        layer.trainable = False
    # print final model configuration after fine-tuning VGG16
    return my_model


# https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    #inputShape = (height * width * depth)
    chanDim = -1

    # define the model input
    # inputs = tf.keras.Input(shape=(None, 1), name='image')
    inputs = tf.keras.Input(shape=inputShape, name='image')
    print("777777777777777")
    print(type(inputs))
    # print(inputs.shape)
    print(inputs)
    # inputs = tf.reshape(inputs, tf.pack([IMG_WIDTH, IMG_HEIGHT, 3]))

    # Load an color image in grayscale\
    # img = cv2.imread('../../data/filtered_images_for_training/no-damage\\2d6341c5-299b-4834-aca9-bfead35d4c83_no-damage_hurricane-harvey_00000008_post_disaster.png', 1)
    # print(img.shape)
    # inputs[0] = img
    # cur_layer = inputs

    # Reshape the image
    # x =

    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs
            # x = tf.keras.layers.Reshape(inputShape, input_shape=(1,))(inputs)
            print("**************")
            # print(x.shape)
        # CONV => RELU => BN => POOL
        x = tf.keras.layers.Conv2D(f, (3, 3), padding="same")(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(16)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = tf.keras.layers.Dense(4)(x)
    x = tf.keras.layers.Activation("relu")(x)

    # check to see if the regression node should be added
    if regress:
        x = tf.keras.layers.Dense(1, activation="linear")(x)

    # construct the CNN
    model = tf.keras.Model(inputs, x)

    # return the CNN
    return model


def make_mlp_functional_model(hparams):
    """Creates a functional API-based multi-layer perceptron model."""
    inputs = tf.keras.Input(
        # shape=(hparams.max_seq_length,), dtype='string', name='image')
        shape=(1,), dtype='string', name='image')

    # JDW
    # # Input is already one-hot encoded in the integer format. We cast it to
    # # floating point format here.
    cur_layer = tf.keras.layers.Lambda(
        lambda x: tf.keras.backend.cast(x, tf.float32))(
        inputs)
    # cur_layer = tf.image.decode_image(inputs)

    for num_units in hparams.num_fc_units:
        # cur_layer = tf.keras.layers.Dense(num_units, activation='relu')(cur_layer)
        cur_layer = tf.keras.layers.Dense(num_units, activation='relu')(cur_layer)
        # For functional models, by default, Keras ensures that the 'dropout' layer
        # is invoked only during training.
        cur_layer = tf.keras.layers.Dropout(hparams.dropout_rate)(cur_layer)

    outputs = tf.keras.layers.Dense(
        hparams.num_classes, activation='softmax')(
        cur_layer)

    model = tf.keras.Model(inputs, outputs=outputs)
    return model


def input_fn(is_training, filenames, batch_size, num_epochs=1, num_parallel_calls=1):
    dataset = tf.data.TFRecordDataset(filenames)
    if is_training:
        dataset = dataset.shuffle(buffer_size=1500)
    dataset = dataset.map(lambda value: parse_record(value, is_training),
                          num_parallel_calls=num_parallel_calls)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def train_input_fn(file_path):
    return input_fn(True, file_path, 100, None, 10)


def validation_input_fn(file_path):
    return input_fn(False, file_path, 50, 1, 1)

def _tester(image_filename):
    #print(image_filename)
    #image = cv2.imread(
    #'../../data/filtered_images_for_training/no-damage\\2d6341c5-299b-4834-aca9-bfead35d4c83_no-damage_hurricane-harvey_00000008_post_disaster.png',
    #1)
    print("]]]]]]]]]]]]]]" + str(image_filename))
    image = cv2.imread(
        '../../data/filtered_images_for_training/no-damage\\2d6341c5-299b-4834-aca9-bfead35d4c83_no-damage_hurricane-harvey_00000008_post_disaster.png',
    1)
    #file_contents = tf.io.read_file(image_filename)
    #image = tf.image.decode_png(file_contents, channels=3)
    #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    #print(image.shape)
    print(type(image))
    print(image.shape)
    print("]]]]]]]]]]]]]]]]")
    return image

def _augment_helper(image_filename):
    """
    Note: Everything in here needs to be a tensor method.
    :param image_filename:
    :return:
    """
    #print(image_filename)
    #image = cv2.imread(
    #'../../data/filtered_images_for_training/no-damage\\2d6341c5-299b-4834-aca9-bfead35d4c83_no-damage_hurricane-harvey_00000008_post_disaster.png',
    #1)
    tf.print(image_filename)
    file_contents = tf.io.read_file(image_filename)
    image = tf.image.decode_png(file_contents, channels=3)
    return image

def parse_fn(example):
  "Parse TFExample records and perform simple data augmentation."
  example_fmt = {
    "image_filename": tf.io.FixedLenFeature([], tf.string, ""),
    #"image": tf.io.FixedLenFeature([], tf.string, ""),
    "label": tf.io.FixedLenFeature([], tf.int64, -1)
  }
  parsed = tf.io.parse_single_example(example, example_fmt)
  #image = tf.io.decode_image(parsed["image"])
  #image = tf.image.decode_png(parsed["image"], channels=3)
  #image = tf.io.decode_raw(parsed['image'], tf.uint8)
  #image_shape = tf.pack([1024, 1024, 3])
  #image = tf.reshape(image, image_shape)
  image = _augment_helper(parsed["image_filename"])  # augments image using slice, reshape, resize_bilinear
  return image, parsed["label"]

def make_dataset_3(file_path, training=False):
  files = tf.data.Dataset.list_files(file_path)
  dataset = tf.data.TFRecordDataset(files)
  #dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
  dataset = dataset.map(map_func=parse_fn)
  dataset = dataset.batch(batch_size=HPARAMS.batch_size)
  return dataset

# Global (cant seem to pas this in correctly.
feature_spec = {}
IMG_HEIGHT = 1024
IMG_WIDTH = 1024
DEPTH = 3
if __name__ == '__main__':
    # Resets tensorflow state
    tf.keras.backend.clear_session()

    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")
    print("Easer Execution", str(tf.executing_eagerly()))

    TEST_DATA_PATH = '../../data/filtered_images_for_training_tf/test.tfr'
    TRAIN_DATA_PATH = '../../data/filtered_images_for_training_tf/train.tfr'

    # Constants used to identify neighbor features in the input.
    NBR_FEATURE_PREFIX = 'NL_nbr_'
    NBR_WEIGHT_SUFFIX = '_weight'

    # Create hyper parameters
    HPARAMS = HParams()

    print("Creating training dataset...")
    #train_dataset = make_dataset(TRAIN_DATA_PATH, training=True)

    train_dataset = make_dataset_3(TRAIN_DATA_PATH, training=True)
    # train_dataset = make_dataset2(TRAIN_DATA_PATH, training=True)

    # print("Creating test dataset...")
    test_dataset = make_dataset_3(TEST_DATA_PATH)

    # Create a base MLP model using the functional API.
    # Alternatively, you can also create a sequential or subclass base model using
    # the make_mlp_sequential_model() or make_mlp_subclass_model() functions
    # respectively, defined above. Note that if a subclass model is used, its
    # summary cannot be generated until it is built.
    print("Creating base model...")
    # base_model_tag, base_model = 'FUNCTIONAL', make_mlp_functional_model(HPARAMS)
    # base_model.summary()

    # base_model = simple_model()
    # base_model.summary()

    #base_model = create_cnn(IMG_WIDTH, IMG_HEIGHT, DEPTH)
    #base_model.summary()

    # create my new model, setting outout as 'x'
    base_model = create_vgg()
    base_model.summary()

    print("Compiling and training base model...")
    base_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    print("Fitting base model with training data...")
    # TODO CHANG ETHIS!!!!!
    print("START START START STATRT")
    base_model.fit(train_dataset, epochs=HPARAMS.epochs, verbose=1)

    # Helper function to print evaluation metrics.
    eval_results = dict(
        zip(base_model.metrics_names,
            base_model.evaluate(test_dataset, steps=HPARAMS.eval_steps)))
    print_metrics('Base MLP model', eval_results)

    print("Creating new base model...")
    base_reg_model_tag, base_reg_model = 'FUNCTIONAL', make_mlp_functional_model(
        HPARAMS)

    print("Adding graph regularization...")
    graph_reg_config = nsl.configs.make_graph_reg_config(
        max_neighbors=HPARAMS.num_neighbors,
        multiplier=HPARAMS.graph_regularization_multiplier,
        distance_type=HPARAMS.distance_type,
        sum_over_axis=-1)
    graph_reg_model = nsl.keras.GraphRegularization(base_reg_model,
                                                    graph_reg_config)
    print("Compiling and training graph base model...")
    graph_reg_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    graph_reg_model.fit(train_dataset, epochs=HPARAMS.train_epochs, verbose=1)

    eval_results = dict(
        zip(graph_reg_model.metrics_names,
            graph_reg_model.evaluate(test_dataset, steps=HPARAMS.eval_steps)))
    print_metrics('MLP + graph regularization', eval_results)

# dataset = dataset.map(lambda value: parse_record(value, is_training),
#                         num_parallel_calls=num_parallel_calls)

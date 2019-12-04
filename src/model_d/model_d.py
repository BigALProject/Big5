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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

import neural_structured_learning as nsl


class HParams(object):
    def __init__(self):
        # set image size
        self.input_shape = [IMG_HEIGHT, IMG_WIDTH, DEPTH]

        # compute length
        self.max_seq_length = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]

        ### neural graph learning parameters
        # Note some items had only 7 neighbors, not sure why, so lets use 7.
        self.num_neighbors = 3
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
        tf.keras.layers.MaxPooling2D(pool_size=HPARAMS.pool_size, strides=2, padding='valid',
                                     data_format='channels_last'),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=HPARAMS.pool_size, strides=2, padding='valid',
                                     data_format='channels_last'),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=HPARAMS.pool_size, strides=2, padding='valid',
                                     data_format='channels_last'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='softmax')
    ])
    return model


def create_vgg():
    """
    This is the same model VONG used in his demo.
    :return:  model
    """
    # Load the VGG16 model
    # resize default image size to 128x128 to speed up model
    input_shape = (IMG_HEIGHT, IMG_WIDTH, DEPTH)
    inputs = tf.keras.Input(shape=input_shape, name='image')
    vgg_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False,
                                            input_tensor=inputs)
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


def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
    """
    Example CNN model.
    References:  https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
    :param width: Image width
    :param height: Image Height
    :param depth: Image Depth
    :param filters: Filter layers
    :param regress: Not sure what this is.
    :return: model
    """
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1

    # define the model input
    # inputs = tf.keras.Input(shape=(None, 1), name='image')
    inputs = tf.keras.Input(shape=inputShape, name='image')

    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs
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


def _augment_helper(image_filename):
    """
    Note: Everything in here needs to be a tensor method.
    :param image_filename:
    :return:
    """
    # In some cases, the image is not filled in.  Use default image.
    if "DEFAULT" == image_filename:
        image_filename = "../../data/filtered_images_for_training/destroyed\\\\0dcb5dc3-dc08-4a7d-8541-3b0cfb2dca91_destroyed_hurricane-harvey_00000072_post_disaster.png"
        tf.print("*************************************************************")
        tf.print("***************** WARNING: DEFAULT IMAGE ********************")
        tf.print("*************************************************************")
    file_contents = tf.io.read_file(image_filename)
    image = tf.image.decode_png(file_contents, channels=3)
    # Required or graph model complains.
    image = tf.cast(image, tf.float32)
    return image


def parse_fn(example):
    "Parse TFExample records and perform simple data augmentation."
    example_fmt = {
        "image_filename": tf.io.FixedLenFeature([], tf.string, ""),
        # "image": tf.io.FixedLenFeature([], tf.string, ""),
        "label": tf.io.FixedLenFeature([], tf.int64, -1),
    }

    neighbor_list = []
    for i in range(HPARAMS.num_neighbors):
        nbr_image_filename = '{}{}_{}'.format(NBR_FEATURE_PREFIX, i, 'image_filename')
        nbr_feature_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX, i, 'image')
        nbr_weight_key = '{}{}{}'.format(NBR_FEATURE_PREFIX, i, NBR_WEIGHT_SUFFIX)

        # Add entry for neighborhood images
        example_fmt[nbr_image_filename] = tf.io.FixedLenFeature([], tf.string, default_value='DEFAULT')

        # Adding neighborhood image file paths to list for later loading.
        neighbor_list.append({'image_filename': nbr_image_filename, 'feature_name': nbr_feature_key})

        # We assign a default value of 0.0 for the neighbor weight so that
        # graph regularization is done on samples based on their exact number
        # of neighbors. In other words, non-existent neighbors are discounted.
        example_fmt[nbr_weight_key] = tf.io.FixedLenFeature(
            [1], tf.float32, default_value=tf.constant([0.0]))

    # Parse entry
    parsed = tf.io.parse_single_example(example, example_fmt)

    # Use this if image is included in entry.  I removed because the file was large and it didn't reduce time.
    # image = tf.io.decode_image(parsed["image"])
    # image = tf.image.decode_png(parsed["image"], channels=3)
    # image = tf.io.decode_raw(parsed['image'], tf.uint8)
    # image_shape = tf.pack([1024, 1024, 3])
    # image = tf.reshape(image, image_shape)

    # Load the main image and add to the resulting dictionary.
    image = _augment_helper(parsed["image_filename"])  # augments image using slice, reshape, resize_binary
    parsed['image'] = image

    # Add all the images
    for neighborhood in neighbor_list:
        tf.print("  Reading neighborhood image=" + parsed[neighborhood['image_filename']] + " into " + str(neighborhood))
        image = _augment_helper(parsed[neighborhood['image_filename']])
        parsed[neighborhood['feature_name']] = image
    labels = parsed.pop('label')
    return parsed, labels


def make_dataset(file_path, training=False):
    """
        Read in the file and generate tensor that includes the image and label.

    :param file_path: Image to read.
    :param training: Flat to specify training or test data.
    :return: Model
    """
    files = tf.data.Dataset.list_files(file_path)
    dataset = tf.data.TFRecordDataset(files)
    # if training:
    #    dataset = dataset.shuffle(buffer_size=1000)
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

    run_base_model = False

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
    train_dataset = make_dataset(TRAIN_DATA_PATH, training=True)

    # print("Creating test dataset...")
    test_dataset = make_dataset(TEST_DATA_PATH)

    print("Creating base model...")

    # Choose a model
    # base_model_tag, base_model = 'FUNCTIONAL', make_mlp_functional_model(HPARAMS)
    # base_model.summary()

    # base_model = simple_model()
    # base_model.summary()

    # base_model = create_cnn(IMG_WIDTH, IMG_HEIGHT, DEPTH)
    # base_model.summary()

    # Create same model as Vong's model
    if run_base_model:
        base_model = create_vgg()
        base_model.summary()

        print("Compiling and training base model...")
        base_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        print("Fitting base model with training data...")
        base_model.fit(train_dataset, epochs=HPARAMS.epochs, verbose=1)

        # Helper function to print evaluation metrics.
        eval_results = dict(
            zip(base_model.metrics_names,
                base_model.evaluate(test_dataset, steps=HPARAMS.eval_steps)))
        print_metrics('Base model', eval_results)

    print("Creating new base model...")
    base_reg_model = create_vgg()

    print("Adding graph regularization...")
    graph_reg_config = nsl.configs.make_graph_reg_config(
        max_neighbors=HPARAMS.num_neighbors,
        multiplier=HPARAMS.graph_regularization_multiplier,
        distance_type=HPARAMS.distance_type,
        sum_over_axis=-1)
    graph_reg_model = nsl.keras.GraphRegularization(base_reg_model,
                                                    graph_reg_config)

    print("Compiling graph model...")
    graph_reg_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    print("Training graph model...")
    graph_reg_model.fit(train_dataset, epochs=HPARAMS.epochs, verbose=1)

    eval_results = dict(
        zip(graph_reg_model.metrics_names,
            graph_reg_model.evaluate(test_dataset, steps=HPARAMS.eval_steps)))
    print_metrics('Base Model + graph regularization', eval_results)

    print("")
    graph_reg_model.summary()
    print("")
    print("done")
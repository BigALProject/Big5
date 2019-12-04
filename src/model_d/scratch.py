
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

#def _tester(image_filename):
#    #print(image_filename)
#    #image = cv2.imread(
#    #'../../data/filtered_images_for_training/no-damage\\2d6341c5-299b-4834-aca9-bfead35d4c83_no-damage_hurricane-harvey_00000008_post_disaster.png',
#    #1)
#    print("]]]]]]]]]]]]]]" + str(image_filename))
#    image = cv2.imread(
#        '../../data/filtered_images_for_training/no-damage\\2d6341c5-299b-4834-aca9-bfead35d4c83_no-damage_hurricane-harvey_00000008_post_disaster.png',
#    1)
#    #file_contents = tf.io.read_file(image_filename)
#    #image = tf.image.decode_png(file_contents, channels=3)
#    #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
#    #print(image.shape)
#    print(type(image))
#    print(image.shape)
#    print("]]]]]]]]]]]]]]]]")
#    return image

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


def augment(x, y):
    x['image'] = cv2.imread(
        '../../data/filtered_images_for_training/no-damage\\2d6341c5-299b-4834-aca9-bfead35d4c83_no-damage_hurricane-harvey_00000008_post_disaster.png',
        1)
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


def make_mlp_functional_model(hparams):
    """Creates a functional API-based multi-layer perceptron model."""
    inputs = tf.keras.Input(
        shape=(hparams.max_seq_length,), dtype='string', name='image')

    # JDW

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

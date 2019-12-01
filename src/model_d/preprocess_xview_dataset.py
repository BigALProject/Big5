"""
  preprocess_xview_dataset.py
  SEIS764 - Artificial Intelligence
  John Wagener, Kwame Sefah, Gavin Pedersen, Vong Vang, Zach Aaberg

  This is largely Google code.  We updated where needed to get it to work with our
  images.

  Next: preprocess_xview_dataset_test_parse.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Tool that preprocesses XView data for Graph Keras trainers.

In particular, this tool does the following:
(a) Converts XView data (xview.content) into TF Examples,
(b) Parses the XView neighborhood graph (xview.neighborhood),
(c) Merges/combines the TF Examples and the graph, and
(d) Writes the training and test data in TF Record format.

The 'xview.content' has the following TSV format:

  house_id<TAB>image path<TAB>destruction_label

Each line of xview.content is:
- Has an integer 'house_id'
- Described by a image of the destruction.
- Has a string 'label' representing the destruction category.

The 'xview.neighborhood' is a TSV file that specifies a graph as a set of edges
representing relationships among houses. 'xview.neighborhood' has the
following TSV format:

  source_house_id<TAB>neighbor_house_id<TAB>distance (weight)

Each line of xview.neighborhood represents an edge that 'source_house_id' has a neighbor
'neighbor_house_id'.

This tool first converts all the 'xview.neighborhood' into TF Examples. Then for
training data, this tool merges into each labeled Example the features of that
Example's neighbors according to that instance's edges in the graph. Finally,
the merged training examples are written to a TF Record file. The test data
will be written to a TF Record file w/o joining with the neighbors.

Sample usage:

$ python preprocess_xview_dataset.py --max_nbrs=5
"""
import collections
import random
import time

import numpy as np
from PIL import Image
from absl import app
from absl import flags
from absl import logging
from neural_structured_learning.tools import graph_utils
import six
import tensorflow as tf

# JDW Character to split file on
split_char = '\t'

FLAGS = flags.FLAGS
FLAGS.showprefixforinfo = False

flags.DEFINE_string(
    'input_xview_content', '../../data/filtered_images_for_training_tf/xview.content',
    """Input file for Xview content that contains ID, image and labels.""")
flags.DEFINE_string('input_xview_graph', '../../data/filtered_images_for_training_tf/xview.neighborhood',
                    """Input file for XView neighborhood graph in TSV format.""")
# JDW Statically set the max_nbrs.  I set to 4 because there are some missing images.
flags.DEFINE_integer(
    'max_nbrs', 1,
    'The maximum number of neighbors to merge into each labeled Example.')
flags.DEFINE_float(
    'train_percentage', 0.8,
    """The percentage of examples to be created as training data. The rest
    are created as test data.""")
flags.DEFINE_string(
    'output_train_data', '../../data/filtered_images_for_training_tf/train.tfr',
    """Output file for training data merged with graph in TF Record format.""")
flags.DEFINE_string('output_test_data', '../../data/filtered_images_for_training_tf/test.tfr',
                    """Output file for test data in TF Record format.""")


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def parse_xview_content(in_file, train_percentage):
    """Converts the Xview content (in TSV) to `tf.train.Example` instances.

    This function parses Xview content (in TSV), converts string labels to integer
    label IDs, randomly splits the data into training and test sets, and returns
    the training and test sets as outputs.

    Args:
      in_file: A string indicating the input file path.
      train_percentage: A float indicating the percentage of training examples
        over the dataset.

    Returns:
      train_examples: A dict with keys being example IDs (string) and values being
      `tf.train.Example` instances.
      test_examples: A dict with keys being example IDs (string) and values being
      `tf.train.Example` instances.
    """
    # JDW: Provides a mapping from string labels to integer indices.
    label_index = {
        'destroyed': 0,
        'no-damage': 1,
        'major-damage': 2,
        'minor-damage': 3,
    }
    # Fixes the random seed so the train/test split can be reproduced.
    random.seed(1)
    train_examples = {}
    test_examples = {}
    counter = 0;
    with open(in_file, 'rU') as xview_content:
        for line in xview_content:
            if counter % 100 == 0:
                print("Processed=" + str(counter))
            counter = counter + 1
            entries = line.rstrip('\n').split(split_char)
            # entries contains [ID, image] are 0-255 values.
            # print(entries)
            image = np.array(Image.open(entries[1]))
            image_raw = image.tostring()
            # JDW Added these features
            xview_content_features = {
                #'image_x': _int64_feature(image.shape[0]),
                #'image_y': _int64_feature(image.shape[1]),
                #'image_filename': _bytes_feature(entries[1].encode('utf-8')),
                'label': _int64_feature(label_index[entries[-1]]),
                # JDW
                #'image': _bytes_feature(image_raw),
                'image': _bytes_feature(entries[1].encode('utf-8')),
            }
            example_features = tf.train.Example(features=tf.train.Features(feature=xview_content_features))

            # JDW: Verify image size, report errors if something doesnt match.
            if image.shape[0] != 1024 or image.shape[1] != 1024:
                raise Exception("Image size is incorrect")
            if 3145728 != len(image_raw):
                raise Exception("Image size is incorrect")

            # Save the house_id to the train or test dataset.
            example_id = entries[0]
            # TODO JDW Need to balance this better.
            if random.uniform(0, 1) <= train_percentage:  # for train/test split.
                train_examples[example_id] = example_features
            else:
                test_examples[example_id] = example_features

        print("Processed=" + str(counter))
    return train_examples, test_examples


def _join_examples(seed_exs, nbr_exs, graph, max_nbrs):
    r"""Joins the `seeds` and `nbrs` Examples using the edges in `graph`.

    This generator joins and augments each labeled Example in `seed_exs` with the
    features of at most `max_nbrs` of the seed's neighbors according to the given
    `graph`, and yields each merged result.

    Args:
      seed_exs: A `dict` mapping node IDs to labeled Examples.
      nbr_exs: A `dict` mapping node IDs to unlabeled Examples.
      graph: A `dict`: source -> (target, weight).
      max_nbrs: The maximum number of neighbors to merge into each seed Example,
        or `None` if the number of neighbors per node is unlimited.

    Yields:
      The result of merging each seed Example with the features of its neighbors,
      as described by the module comment.
    """
    # A histogram of the out-degrees of all seed Examples. The keys of this dict
    # range from 0 to 'max_nbrs' (inclusive) if 'max_nbrs' is finite.
    out_degree_count = collections.Counter()

    def has_ex(node_id):
        """Returns true iff 'node_id' is in the 'seed_exs' or 'nbr_exs dict'."""
        result = (node_id in seed_exs) or (node_id in nbr_exs)
        if not result:
            logging.warning('No tf.train.Example found for edge target ID: "%s"',
                            node_id)
        return result

    def lookup_ex(node_id):
        """Returns the Example from `seed_exs` or `nbr_exs` with the given ID."""
        return seed_exs[node_id] if node_id in seed_exs else nbr_exs[node_id]

    def join_seed_to_nbrs(seed_id):
        """Joins the seed with ID `seed_id` to its out-edge graph neighbors.

        This also has the side-effect of maintaining the `out_degree_count`.

        Args:
          seed_id: The ID of the seed Example to start from.

        Returns:
          A list of (nbr_wt, nbr_id) pairs (in decreasing weight order) of the
          seed Example's top `max_nbrs` neighbors. So the resulting list will have
          size at most `max_nbrs`, but it may be less (or even empty if the seed
          Example has no out-edges).
        """
        nbr_dict = graph[seed_id] if seed_id in graph else {}
        nbr_wt_ex_list = [(nbr_wt, nbr_id)
                          for (nbr_id, nbr_wt) in six.iteritems(nbr_dict)
                          if has_ex(nbr_id)]
        result = sorted(nbr_wt_ex_list, reverse=True)[:max_nbrs]
        out_degree_count[len(result)] += 1

        # JDW Verify there are enough entries here.
        # Downstream is picky about getting full datasets.
        # We could pad this with empty images if we wanted.
        if len(result) < FLAGS.max_nbrs:
            raise Exception("ERROR: less than " + str(FLAGS.max_nbrs) + " --> " + seed_id)
        return result

    def merge_examples(seed_ex, nbr_wt_ex_list):
        """Merges neighbor Examples into the given seed Example `seed_ex`.

        Args:
          seed_ex: A labeled Example.
          nbr_wt_ex_list: A list of (nbr_wt, nbr_id) pairs (in decreasing nbr_wt
            order) representing the neighbors of 'seed_ex'.

        Returns:
          The Example that results from merging the features of the neighbor
          Examples (as well as creating a feature for each neighbor's edge weight)
          into `seed_ex`. See the `join()` description above for how the neighbor
          features are named in the result.
        """
        # Make a deep copy of the seed Example to augment.
        merged_ex = tf.train.Example()
        merged_ex.CopyFrom(seed_ex)

        # Add a feature for the number of neighbors.
        # JDW Disabled this
        #merged_ex.features.feature['NL_num_nbrs'].int64_list.value.append(
        #    len(nbr_wt_ex_list))

        # Enumerate the neighbors, and merge in the features of each.
        for index, (nbr_wt, nbr_id) in enumerate(nbr_wt_ex_list):
            prefix = 'NL_nbr_{}_'.format(index)
            # Add the edge weight value as a new singleton float feature.
            weight_feature = prefix + 'weight'
            merged_ex.features.feature[weight_feature].float_list.value.append(nbr_wt)
            # Copy each of the neighbor Examples features, prefixed with 'prefix'.
            nbr_ex = lookup_ex(nbr_id)
            for (feature_name, feature_val) in six.iteritems(nbr_ex.features.feature):
                # JDW: Fill process both label an image
                print(prefix + feature_name)
                print(feature_val)
                new_feature = merged_ex.features.feature[prefix + feature_name]
                new_feature.CopyFrom(feature_val)

        print("------------------------\n" + str(merged_ex) + "oooooooooooooooooooooooo\n")
        return merged_ex

    start_time = time.time()
    logging.info(
        'Joining seed and neighbor tf.train.Examples with graph edges...')
    for (seed_id, seed_ex) in six.iteritems(seed_exs):
        yield merge_examples(seed_ex, join_seed_to_nbrs(seed_id))
    logging.info(
        'Done creating and writing %d merged tf.train.Examples (%.2f seconds).',
        len(seed_exs), (time.time() - start_time))
    logging.info('Out-degree histogram: %s', sorted(out_degree_count.items()))


def main(unused_argv):
    start_time = time.time()

    # Parses Xview content into TF Examples.
    print("Reading FLAGS.train_percentage=" + str(FLAGS.train_percentage) +
          " from FLAGS.input_xview_content=" + FLAGS.input_xview_content)
    train_examples, test_examples = parse_xview_content(FLAGS.input_xview_content,
                                                        FLAGS.train_percentage)

    print("Reading graph from FLAGS.input_xview_graph=" + FLAGS.input_xview_graph)
    graph = graph_utils.read_tsv_graph(FLAGS.input_xview_graph)
    graph_utils.add_undirected_edges(graph)

    # Joins 'train_examples' with 'graph'. 'test_examples' are used as *unlabeled*
    # neighbors for transductive learning purpose. In other words, the labels of
    # test_examples are not used.
    with tf.io.TFRecordWriter(FLAGS.output_train_data) as writer:
        for merged_example in _join_examples(train_examples, test_examples, graph,
                                             FLAGS.max_nbrs):
            writer.write(merged_example.SerializeToString())

    logging.info('Output training data written to TFRecord file: %s.',
                 FLAGS.output_train_data)

    # Writes 'test_examples' out w/o joining with the graph since graph
    # regularization is used only during training, not testing/serving.
    with tf.io.TFRecordWriter(FLAGS.output_test_data) as writer:
        for example in six.itervalues(test_examples):
            writer.write(example.SerializeToString())

    logging.info('Output test data written to TFRecord file: %s.',
                 FLAGS.output_test_data)
    logging.info('Total running time: %.2f minutes.',
                 (time.time() - start_time) / 60.0)


if __name__ == '__main__':
    # Ensures TF 2.0 behavior even if TF 1.X is installed.
    tf.compat.v1.enable_v2_behavior()
    app.run(main)

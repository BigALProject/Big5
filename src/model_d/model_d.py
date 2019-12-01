"""
  model_d.py
  SEIS764 - Artificial Intelligence
  John Wagener, Kwame Sefah, Gavin Pedersen, Vong Vang, Zach Aaberg

  Uses G

  Next: preprocess_xview_dataset_test_parse.py
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
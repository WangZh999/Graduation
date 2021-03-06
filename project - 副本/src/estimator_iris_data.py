# 引入必要的module
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"


class EstimatorIrisData(object):
    def __init__(self):
        # If the training and test sets aren't stored locally, download them.
        if not os.path.exists(IRIS_TRAINING):
            raw = urllib.request.urlopen(IRIS_TRAINING_URL).read()
            with open(IRIS_TRAINING, "wb") as f:
                f.write(raw)

        if not os.path.exists(IRIS_TEST):
            raw = urllib.request.urlopen(IRIS_TEST_URL).read()
            with open(IRIS_TEST, "wb") as f:
                f.write(raw)

        # Load data sets.
        self.training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
            filename=IRIS_TRAINING,
            target_dtype=np.int,
            features_dtype=np.float32)
        self.test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
            filename=IRIS_TEST,
            target_dtype=np.int,
            features_dtype=np.float32)

        # Specify that all features have real-value data
        self.feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

    # Define the training inputs
    def get_train_inputs(self):
        train_x = tf.constant(self.training_set.data)
        train_y = tf.constant(self.training_set.target)
        return train_x, train_y

    # Define the test inputs
    def get_test_inputs(self):
        test_x = tf.constant(self.test_set.data)
        test_y = tf.constant(self.test_set.target)
        return test_x, test_y

    def estimator(self, hidden_units=None, train_steps=100, n_classes=3, data_set=True):
        """

        :param hidden_units:
        :param train_steps:
        :param n_classes:
        :param data_set:
        :return:
        """

        if hidden_units is None:
            hidden_units = [6, 6]

        print("hidden units: ",hidden_units,"    train steps: ",train_steps)

        classifier = tf.contrib.learn.DNNClassifier(feature_columns=self.feature_columns,
                                                    hidden_units=hidden_units,
                                                    n_classes=n_classes)

        # Fit model.
        classifier.fit(input_fn=self.get_train_inputs, steps=train_steps)

        # Evaluate accuracy.
        if data_set:
            accuracy_score = classifier.evaluate(input_fn=self.get_train_inputs, steps=1)
            return accuracy_score["loss"]
        else:
            accuracy_score = classifier.evaluate(input_fn=self.get_test_inputs, steps=1)
            return accuracy_score["loss"]
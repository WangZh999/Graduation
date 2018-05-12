from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf

from iris_data import IrisData

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')


class Estimator(object):
    def __init__(self, hidden_units=[], train_steps=1000):
        self.hidden_units = hidden_units
        self.train_steps = train_steps

    def eval(self):
        # Fetch the data
        iris_data = IrisData()
        # (train_x, train_y), (test_x, test_y) = iris_data.load_data()

        # Feature columns describe how to use the input.
        # my_feature_columns = []
        # for key in iris_data.train_x.keys():
        #     my_feature_columns.append(tf.feature_column.numeric_column(key=key))

        # Specify that all features have real-value data
        feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

        # Build 2 hidden layer DNN with 10, 10 units respectively.
        classifier = tf.contrib.learn.DNNClassifier(
            feature_columns=feature_columns,
            # Two hidden layers of 10 nodes each.
            hidden_units=self.hidden_units,
            # The model must choose between 3 classes.
            n_classes=3)

        # Train the Model.
        classifier.fit(
            input_fn=lambda: iris_data.train_input_fn(),
            steps=self.train_steps)

        # Evaluate the model.
        eval_result = classifier.evaluate(
            input_fn=lambda: iris_data.eval_input_fn())

        # print('\nTest set accuracy: {accuracy:0.3f}    loss: {loss:0.3f}
        #  average_loss: {average_loss:0.3f}\n'.format(**eval_result))
        return eval_result

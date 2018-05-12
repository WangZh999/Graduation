# encoding = utf-8

import os

import tensorflow as tf


def environment_init():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

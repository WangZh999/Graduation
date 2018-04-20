import os

import tensorflow as tf

from estimator import Estimator

if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)
    test = Estimator(hidden_units=[10, 10, 10])
    tf.app.run(test.eval)

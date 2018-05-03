import os

import tensorflow as tf

from estimator_iris_data import EstimatorIrisData

if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)
    test = EstimatorIrisData()
    # print(tf.app.run(test.eval))
    print(test.estimator([10, 10]))

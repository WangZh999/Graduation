import pandas as pd
import tensorflow as tf

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


class IrisData(object):
    def __init__(self, y_name='Species'):
        """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
        train_path, test_path = self.maybe_download()

        train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
        self.train_x, self.train_y = train, train.pop(y_name)

        test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
        self.test_x, self.test_y = test, test.pop(y_name)

    def train_input_fn(self, batch_size, features=None, labels=None):
        """An input function for training"""
        if features is None or labels is None:
            features = self.train_x
            labels = self.train_y

        # Convert the inputs to a Data set.
        data_set = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffle, repeat, and batch the examples.
        data_set = data_set.shuffle(1000).repeat().batch(batch_size)

        # Return the data_set.
        return data_set

    def eval_input_fn(self, batch_size, features=None, labels=None):
        """An input function for evaluation or prediction"""
        if features is None or labels is None:
            features = self.test_x
            labels = self.test_y

        features = dict(features)
        if labels is None:
            # No labels, use only features.
            inputs = features
        else:
            inputs = (features, labels)

        # Convert the inputs to a Data set.
        data_set = tf.data.Dataset.from_tensor_slices(inputs)

        # Batch the examples
        assert batch_size is not None, "batch_size must not be None"
        data_set = data_set.batch(batch_size)

        # Return the data_set.
        return data_set

    @staticmethod
    def maybe_download():
        train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL,
                                             cache_dir="F:\\graduation project\\my_work\\project\\iris_data")
        test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL,
                                            cache_dir="F:\\graduation project\\my_work\\project\\iris_data")

        return train_path, test_path


# The remainder of this file contains a simple example of a csv parser,
#     implemented using a the `Dataset` class.

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]


def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Species')

    return features, label


def csv_input_fn(csv_path, batch_size=100):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

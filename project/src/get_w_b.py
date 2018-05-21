# coding: utf-8

import os

import numpy as np
import tensorflow as tf

IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

if not os.path.exists(IRIS_TRAINING):
    raw = urllib.request.urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, "wb") as f:
        f.write(raw)

if not os.path.exists(IRIS_TEST):
    raw = urllib.request.urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST, "wb") as f:
        f.write(raw)

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)

train_x = tf.constant(training_set.data)
train_y = tf.constant(training_set.target)
test_x = tf.constant(test_set.data)
test_y = tf.constant(test_set.target)

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 4])
y = tf.placeholder(tf.float32, [None, 3])

# 创建一个简单的神经网络
W1 = tf.Variable(tf.truncated_normal([4, 1], stddev=0.1))
b1 = tf.Variable(tf.zeros([1]))
L1 = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.truncated_normal([1, 3], stddev=0.1))
b2 = tf.Variable(tf.zeros([3]))
prediction = tf.nn.softmax(tf.matmul(L1, W2) + b2)

# 交叉熵代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# 训练
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    t_x, t_y_t = sess.run([train_x, train_y])
    t_y = []
    for i in t_y_t:
        if i == 0:
            t_y += [[1, 0, 0]]
        elif i == 1:
            t_y += [[0, 1, 0]]
        else:
            t_y += [[0, 0, 1]]

    #     print(t_x,t_y)
    for epoch in range(50):
        sess.run(train_step, feed_dict={x: t_x, y: t_y})
        print(sess.run([W1, b1, W2, b2]))

import random

import tensorflow as tf

_IN_DIM = 4
_OUT_DIM = 1


class GeneticNetwork(object):
    def __init__(self, layer_dimensions=[]):

        self.x = tf.placeholder(tf.float32, [None, _IN_DIM])
        self.y = tf.placeholder(tf.float32, [None, _OUT_DIM])

        previous_out = self.x

        for idx in range(len(layer_dimensions)):
            current_layer = layer_dimensions[idx]
            this_w = tf.Variable(tf.random_normal([_IN_DIM, current_layer]))
            this_b = tf.Variable(tf.zeros([current_layer]))
            wx_plus_b = tf.matmul(previous_out, this_w) + this_b
            previous_out = tf.nn.tanh(wx_plus_b)

        self.predictedOutput = previous_out
        loss = tf.reduce_mean(tf.square(self.predictedOutput - self.y))
        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        init = tf.initialize_all_variables()
        self.sess = tf.Session(config=tf.ConfigProto(
            inter_op_parallelism_threads=4
        ))
        self.sess.run(init)

    def train_step_f(self, ins=[], outs=[], choose_count=-1):
        if choose_count == -1:
            batch_xs, batch_ys = ins, outs
        else:
            chosen_indices = [random.randint(0, len(ins) - 1) for _ in range(choose_count)]
            batch_xs, batch_ys = list(zip(*[(ins[x], [outs[x]]) for x in chosen_indices]))

        _loss = tf.reduce_mean(tf.square(self.predictedOutput - self.y))

        _, loss = self.sess.run([self.train_step, _loss], feed_dict={self.x: batch_xs, self.y: batch_ys})
        return loss

    def close(self):
        self.sess.close()

# encoding: utf-8

import os
import random

import numpy as np
import tensorflow as tf

from estimator_iris_data import EstimatorIrisData


class Global(object):
    """
    The problem related parameters and genetic operations
    """

    def __init__(self, max_layer=5, max_note=20, n=100, m=2, train_steps=50):
        """
        初始化
        :param max_layer: 最多层数
        :param n: 个体个数
        :param m: 目标函数个数
        :param max_note: 每层最多的节点数
        """
        if 0 == n % 2:
            self.N = n
        else:
            self.N = n + 1
        self.M = m
        self.max_layer = max_layer
        self.max_note = max_note
        self.train_steps = train_steps
        self.eval = EstimatorIrisData()

    def initialize(self, num=0):
        """
        随机生成初始种群
        :return:初始种群
        """
        if 0 == num:
            num = self.N
        pop_dec = []
        for i in range(num):
            d = random.randint(1, self.max_layer)
            one_dec = [[random.randint(1, self.max_note) for _ in range(d)]]
            pop_dec += one_dec

        return np.array(pop_dec)

    def cost_fun(self, pop_dec):
        """
        根据决策向量计算目标向量
        :param pop_dec: 决策向量
        :return: 目标向量
        """
        n = pop_dec.shape[0]
        pop_obj = np.zeros((n, self.M))
        for i in range(n):
            pop_obj[i, 0] = np.sum(pop_dec[i])
            train_steps = 300 + pop_obj[i, 0] * 40
            # pop_obj[i, 1] = self.eval.estimator(hidden_units=pop_dec[i], train_steps=self.train_steps)
            pop_obj[i, 1] = self.eval.estimator(hidden_units=pop_dec[i], train_steps=train_steps)

        return pop_obj

    def get_offspring(self, pop_dec):
        pop_dec_format = []
        for dec in pop_dec:
            pop_dec_format += [format_dec(dec, self.max_layer)]
        offspring_dec = []
        for i in range(len(pop_dec_format) // 2):
            temp_dec_1, temp_dec_2 = crossover(pop_dec_format[2 * i], pop_dec_format[2 * i + 1], item_max=self.max_note)
            if 0 == np.sum(temp_dec_1):
                temp_dec_1 = init_one_item(self.max_layer, self.max_note)
            if 0 == np.sum(temp_dec_2):
                temp_dec_2 = init_one_item(self.max_layer, self.max_note)

            offspring_dec += [temp_dec_1]
            offspring_dec += [temp_dec_2]

        return np.array(offspring_dec)


def crossover(dec_1, dec_2, item_max=20):
    """

    :param item_max:
    :param dec_1:
    :param dec_2:
    :return:
    """
    dec_length = len(dec_1)
    dec_after_temp_1 = np.zeros(dec_length)
    dec_after_temp_2 = np.zeros(dec_length)
    for i in range(dec_length):
        chose = random.random()
        if chose < 0.5:
            dec_after_temp_1[i] = dec_1[i]
            dec_after_temp_2[i] = dec_2[i]
        else:
            dec_after_temp_1[i] = dec_2[i]
            dec_after_temp_2[i] = dec_1[i]

    mutation_range = item_max // 5
    if 0 == mutation_range:
        mutation_range = 1
    mutation = random.randint(-mutation_range, mutation_range)
    for i in range(len(dec_1)):
        if dec_after_temp_1[i] > 0:
            dec_after_temp_1[i] = (dec_after_temp_1[i] + mutation) % (item_max + 1)
        if dec_after_temp_2[i] > 0:
            dec_after_temp_2[i] = (dec_after_temp_2[i] + mutation) % (item_max + 1)

    dec_after_1 = []
    dec_after_2 = []
    for item1 in dec_after_temp_1:
        if 0 < item1 <= item_max:
            dec_after_1 += [int(item1)]
    for item2 in dec_after_temp_2:
        if 0 < item2 <= item_max:
            dec_after_2 += [int(item2)]

    return dec_after_1, dec_after_2


def format_dec(dec, n):
    """

    :param n:
    :param dec:
    :return:
    """
    dec_length = len(dec)
    if dec_length < n:
        temp_dec = np.zeros(n)
        rand_n = random_n(dec_length, max=n)
        for i in range(dec_length):
            temp_dec[rand_n[i]] = dec[i]
        return temp_dec

    return dec


def random_n(n, max):
    """
    随机产生n个不同的随机数
    :param min_max:
    :param n:
    :return: n个不同的随机数
    """
    result_list = random.sample(range(max), n)
    return sorted(result_list)


def init_one_item(max_col, max_value):
    d = random.randint(1, max_col)
    _item = [random.randint(1, max_value) for _ in range(d)]
    return _item


if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)
    # test = Global(n=5)
    # seed = test.initialize()
    # obj = test.cost_fun(seed)
    # print(obj)
    #
    # t1 = random_n(2, 20)
    # t2 = random_n(3, 20)
    # print(t1, "  ", t2)
    # ft1 = format_dec(t1, 5)
    # ft2 = format_dec(t2, 5)
    # print(ft1, "  ", ft2)
    # ct1, ct2 = test.crossover(ft1, ft2)
    # print(ct1, "  ", ct2)

    test = Global(n=10)
    seed = test.initialize()
    for _ in range(10):
        obj = test.cost_fun(seed)
        print("dec: ", seed)
        print("obj: ", obj)
        seed = test.get_offspring(seed)

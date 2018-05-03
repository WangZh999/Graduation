#!/usr/bin/env python
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

    def __init__(self, max_layer=5, n=100, m=2, max_note=50):
        """
        初始化
        :param max_layer: 最多层数
        :param n: 个体个数
        :param m: 目标函数个数
        :param max_note: 每层最多的节点数
        """
        self.N = n
        self.M = m
        self.max_layer = max_layer
        self.max_note = max_note
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
            print(pop_dec[i])
            pop_obj[i, 0] = np.sum(pop_dec[i])
            pop_obj[i, 1] = self.eval.estimator(hidden_units=pop_dec[i])

        return pop_obj

    def get_offspring(self, pop_dec):
        pop_dec_format = []
        for dec in pop_dec:
            pop_dec_format += [format_dec(dec,self.max_layer)]
        offspring_dec=[]
        for i in range(len(pop_dec_format)//2):
            temp_dec_1,temp_dec_2=self.crossover(pop_dec_format[2*i],pop_dec_format[2*i+1])
            offspring_dec+=[temp_dec_1]
            offspring_dec+=[temp_dec_2]

        return offspring_dec

    def crossover(self, dec_1, dec_2):
        """

        :param dec_1:
        :param dec_2:
        :return:
        """
        dec_after_temp_1 = np.zeros(self.max_layer)
        dec_after_temp_2 = np.zeros(self.max_layer)
        for i in range(len(dec_1)):
            chose = random.randint()
            if chose < 0.5:
                dec_after_temp_1[i] = dec_1[i]
                dec_after_temp_2[i] = dec_2[i]
            else:
                dec_after_temp_1[i] = dec_2[i]
                dec_after_temp_2[i] = dec_1[i]

        if 0 == np.sum(dec_after_temp_1):
            dec_after_temp_1 = [random.randint(1, self.max_note) for _ in range(random.randint(1, self.max_layer))]
        if 0 == np.sum(dec_after_temp_2):
            dec_after_temp_2 = [random.randint(1, self.max_note) for _ in range(random.randint(1, self.max_layer))]

        dec_after_1 = []
        dec_after_2 = []
        for item1 in dec_after_temp_1:
            if item1 > 0:
                dec_after_1 += item1
        for item2 in dec_after_temp_2:
            if item2 > 0:
                dec_after_2 += item2

        bianyi=random.randint(2)
        index=random.randint()


        return dec_after_1, dec_after_2


def format_dec(dec, n):
    """

    :param dec:
    :return:
    """
    dec_length = len(dec)
    if dec_length < n:
        temp_dec = np.zeros(n)
        rand_n = randrom_n(dec_length, min_max=(0, n))
        for i in range(dec_length):
            temp_dec[rand_n[i]] = dec[i]
        return temp_dec

    return dec


def randrom_n(n, min_max):
    """
    随机产生n个不同的随机数
    :param n:
    :return: n个不同的随机数
    """
    return [1, 2, 3, 4]


if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)
    test = Global(n=5)
    seed = test.initialize()
    obj = test.cost_fun(seed)
    print(obj)

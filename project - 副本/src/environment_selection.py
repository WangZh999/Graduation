# encoding: utf-8
import numpy as np

from crowding_distance import crowding_distance
from nd_sort import nd_sort


def environment_selection(pop_dec, pop_obj, n):
    """

    :param pop_dec:decision of population
    :param pop_obj:objective of population
    :param n:num of offspring
    :return:
    """
    pop_num = len(pop_dec)
    front_non, max_front = nd_sort(pop_obj, n)
    next_label = [False for _ in range(front_non.size)]
    for i in range(front_non.size):
        if front_non[i] < max_front:
            next_label[i] = True
    crowd_dis = crowding_distance(pop_obj, front_non, max_front)
    last = [i for i in range(pop_num) if front_non[i] == max_front]
    rank = np.argsort(-crowd_dis[last])
    delta_n = rank[: (n - int(np.sum(next_label)))]
    rest = [last[i] for i in delta_n]
    for i in rest:
        next_label[i] = True
    index = np.array([i for i in range(pop_num) if next_label[i]])
    return pop_dec[index], pop_obj[index]

    # next_pop = [pop_dec[index, :], pop_obj[1][index, :]]
    # return next_pop, front_non[index], crowd_dis[index], index

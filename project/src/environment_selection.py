#!/usr/bin/env python
# encoding: utf-8
import numpy as np

from crowding_distance import crowding_distance
from nd_sort import nd_sort


def environment_selection(population, n):
    '''
    environmental selection in NSGA-II
    :param population: current population
    :param n: number of selected individuals
    :return: next generation population
    '''
    front_no, max_front = nd_sort(population[1], n)
    next_label = [False for i in range(front_no.size)]
    for i in range(front_no.size):
        if front_no[i] < max_front:
            next_label[i] = True
    crowd_dis = crowding_distance(population[1], front_no)
    last = [i for i in range(len(front_no)) if front_no[i] == max_front]
    rank = np.argsort(-crowd_dis[last])
    delta_n = rank[: (n - int(np.sum(next_label)))]
    rest = [last[i] for i in delta_n]
    for i in rest:
        next_label[i] = True
    index = np.array([i for i in range(len(next_label)) if next_label[i]])
    next_pop = [population[0][index, :], population[1][index, :]]
    return next_pop, front_no[index], crowd_dis[index], index

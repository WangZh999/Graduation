# encoding: utf-8

import numpy as np
from matplotlib import pyplot as plt

from environment_init import environment_init
from estimator_iris_data import EstimatorIrisData
from nd_sort import nd_sort
from nsga import NSGA_II


def get_test_obj(pop_dec):
    n = pop_dec.shape[0]
    pop_obj_test = np.zeros((n, 1))
    eval = EstimatorIrisData()
    for i in range(n):
        pop_obj_test[i, 0] = eval.estimator(hidden_units=pop_dec[i],
                                            train_steps=300 + 40 * pop_obj_test[i, 0],
                                            data_set=False)
    return pop_obj_test


def draw(pop_obj):
    front_no, max_front = nd_sort(pop_obj[:, 0:2], 1)
    non_dominated = pop_obj[front_no == 1, :]
    plt.scatter(non_dominated[:, 0], non_dominated[:, 1])
    plt.scatter(non_dominated[:, 0], non_dominated[:, 2])


if __name__ == "__main__":
    environment_init()
    nsga = NSGA_II(iter=20)
    pop_dec, pop_obj = nsga.run()
    front_no, max_front = nd_sort(pop_obj, 1)
    non_dominated_dec = pop_dec[front_no == 1]
    non_dominated_obj_train = pop_obj[front_no == 1]

    non_dominated_obj_test = get_test_obj(non_dominated_dec)
    # draw(non_dominated_obj)
    # plt.show()

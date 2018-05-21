# encoding: utf-8

import numpy as np
from matplotlib import pyplot as plt

from Global import Global
from environment_init import environment_init
from environment_selection import environment_selection
from nd_sort import nd_sort

nsga_global = Global(max_layer=3, max_note=6, n=20, m=2)


class NSGA_II(object):
    def __init__(self, decs=None, iter=100):
        self.decs = decs
        self.iter = iter

    def run(self):
        f = open("out.txt", "w+")
        if self.decs is None:
            parent_dec = nsga_global.initialize()
            parent_obj = nsga_global.cost_fun(parent_dec)
        else:
            parent_dec = self.decs
            parent_obj = nsga_global.cost_fun(parent_dec)

        for _ in range(self.iter):
            print("\n\n", _, "\n\n")

            offspring_dec = nsga_global.get_offspring(parent_dec)
            offspring_obj = nsga_global.cost_fun(offspring_dec)

            pop_dec_all = np.append(parent_dec, offspring_dec, axis=0)
            pop_dec, return_index = np.unique(pop_dec_all, return_index=True)
            pop_obj_all = np.append(parent_obj, offspring_obj, axis=0)
            pop_obj = pop_obj_all[return_index]

            print("dec", _, "\n:", pop_dec, file=f)
            print("obj", _, "\n:", pop_obj, file=f)

            parent_dec, parent_obj = environment_selection(pop_dec=pop_dec, pop_obj=pop_obj, n=nsga_global.N)
        f.close()
        return parent_dec, parent_obj


def draw(pop_obj):
    front_no, max_front = nd_sort(pop_obj, 1)
    non_dominated = pop_obj[front_no == 1, :]
    if nsga_global.M == 2:
        plt.scatter(non_dominated[:, 0], non_dominated[:, 1])
    elif nsga_global.M == 3:
        x, y, z = non_dominated[:, 0], non_dominated[:, 1], non_dominated[:, 2]
        ax = plt.subplot(111, projection='3d')
        ax.scatter(x, y, z, c='b')
    else:
        for i in range(len(non_dominated)):
            plt.plot(range(1, nsga_global.M + 1), non_dominated[i, :])


if __name__ == "__main__":
    environment_init()
    nsga = NSGA_II(iter=20)
    dec, obj = nsga.run()
    # draw(obj)

    # obj = np.array([[1., 1.28706038],
    #                 [2., 0.73423541],
    #                 [5., 0.6547901],
    #                 [3., 1.05101264],
    #                 [7., 1.09628367],
    #                 [4., 0.53830171]])
    #
    # draw(obj)
    # plt.show()

# encoding: utf-8
import numpy as np


def crowding_distance(pop_obj, front_non, front):
    """
    The crowding distance of one Pareto front
    :param fronts: Pareto front
    :param pop_obj: objective vectors
    :param front_non: front numbers
    :return: crowding distance
    """
    n, m = np.shape(pop_obj)
    crowd_dis = np.ones(n) * np.inf
    obj_front = np.array([k for k in range(len(front_non)) if front_non[k] == front])
    obj_max = pop_obj[obj_front, :].max(0)
    obj_min = pop_obj[obj_front, :].min(0)
    for i in range(m):
        rank = np.argsort(pop_obj[obj_front, i])
        crowd_dis[obj_front[rank[0]]] = np.inf
        crowd_dis[obj_front[rank[-1]]] = np.inf
        for j in range(1, len(obj_front) - 1):
            crowd_dis[obj_front[rank[j]]] = crowd_dis[obj_front[rank[j]]] + (
                    pop_obj[(obj_front[rank[j + 1]], i)] - pop_obj[(obj_front[rank[j - 1]], i)]) / (
                                                    obj_max[i] - obj_min[i])
    return crowd_dis

    # fronts = np.unique(front_non)
    # Fronts = fronts[fronts != np.inf]
    # for f in range(len(Fronts)):
    #     Front = np.array([k for k in range(len(front_non)) if front_non[k] == Fronts[f]])
    #     Fmax = pop_obj[Front, :].max(0)
    #     Fmin = pop_obj[Front, :].min(0)
    #     for i in range(m):
    #         rank = np.argsort(pop_obj[Front, i])
    #         crowd_dis[Front[rank[0]]] = np.inf
    #         crowd_dis[Front[rank[-1]]] = np.inf
    #         for j in range(1, len(Front) - 1):
    #             crowd_dis[Front[rank[j]]] = crowd_dis[Front[rank[j]]] + (pop_obj[(Front[rank[j + 1]], i)] - pop_obj[
    #                 (Front[rank[j - 1]], i)]) / (Fmax[i] - Fmin[i])
    # return crowd_dis

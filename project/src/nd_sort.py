import numpy as np


def nd_sort(pop_obj, n_sort):
    """
    :rtype:
    :param n_sort:
    :param pop_obj: objective vectors
    :return: [FrontNon, MaxFNo]
    """
    n, m_obj = np.shape(pop_obj)
    front_non = np.inf * np.ones(n)
    max_front = 0
    while np.sum(front_non < np.inf) < min(n_sort, n):
        max_front += 1
        for i in range(n):
            if front_non[i] == np.inf:
                dominated = False
                for j in range(n):
                    if front_non[j] >= max_front and j != i:
                        m = 0
                        while (m < m_obj) and (pop_obj[i, m] >= pop_obj[j, m]):
                            m += 1
                        dominated = (m >= m_obj)
                        if dominated:
                            break
                if not dominated:
                    front_non[i] = max_front

    return front_non, max_front


if __name__ == "__main__":
    pop_obj = np.array([[0, 9], [0, 8], [5, 4], [3, 5], [6, 5]])
    a, b = nd_sort(pop_obj, 3)

    print(a, b)

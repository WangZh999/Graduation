from matplotlib import pyplot as plt

from environment_init import environment_init

if __name__ == '__main__':
    environment_init()
    decs = [[1],
            [1, 1],
            [2],
            [2, 2],
            [2, 3],
            [3],
            [4],
            [5],
            [5, 5, 4],
            [6, 6]]

    # test = EstimatorIrisData()
    # loss = []
    # for dec in decs:
    #     loss += [test.estimator(dec, train_steps=50)]

    # print(test.estimator([10, 10]))

    loss_test = [
        1.119664700,
        0.620649700,
        0.783948000,
        0.681741200,
        0.810779040,
        0.637001800,
        0.521618800]
    sum_dec = [
        1,
        2,
        3,
        4,
        5,
        12,
        14]
    loss_train = [
        1.13232863,
        0.85842025,
        0.74991554,
        0.6465959,
        0.45460477,
        0.31375214,
        0.29347035]

    plt.scatter(sum_dec, loss_train)
    plt.scatter(sum_dec, loss_test)
    plt.show()

# import sys
# sys.path.append('/anaconda3/lib/python3.6/')


import matplotlib
# Notice: do not remve this!
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


# def plot():
#     x = [1, 2, 3, 4]
#     y = [3, 4, 5, 6]
#
#     plt.figure()
#     plt.plot(x, y)
#     plt.show()



def plot(train_x = [], train_y = [], predict_x = [], predict_y = [], test_x = [], test_y = []):
    plt.figure()

    if train_x != []:
        plot_scatter(train_x, train_y)

    if predict_x != []:
        plt.plot(predict_x, predict_y)

    if test_x != []:
        plot_scatter(test_x, test_y)

    plt.show()

def plot_scatter(x, y):
    for i in range(x.shape[0]):
        if y[i] == 1:
            plt.scatter(x[i][0], x[i][1], marker='+', c = 100)
        else:
            plt.scatter(x[i][0], x[i][1], marker='o', c=100)

# df = np.array([[3,3],
#                [4,3],
#                [1,1]])
# y = np.array([1, 1, -1])
# plot(df, y)
import pandas as pd
import numpy as np


class Perceptron:
    def __init__(self):
        self.step = 1
        self.max_step_count = 50

    def train(self, features, labels):
        step_count = 0
        self.w = np.zeros(features.shape[1])
        self.b = 0
        i = 0
        correct_count = 0
        while step_count <= self.max_step_count:
            tmp = labels[i] * (features[i].dot(self.w) + self.b)
            if tmp <= 0:
                self.w = self.w + self.step * labels[i] * features[i]
                self.b = self.b + self.step * labels[i]
                step_count += 1
                correct_count = 0
            else:
                correct_count += 1
                if correct_count > features.shape[0]:
                    break
                else:
                    i += 1
                    i = i % features.shape[0]
        return (self.w, self.b)

    def train_dual(self, features, labels):
        n = features.shape[0]
        di = features.shape[1]
        gamma = np.zeros([n, n])
        step_count = 0
        self.a = np.zeros(n)
        self.b = 0
        correct_count = 0

        for i in range(n):
            for j in range(n):
                for k in range(di):
                    gamma[i][j] += features[i][k] * features[j][k]

        tmp = 0
        i = 0
        while step_count <= self.max_step_count:
            tmp1 = 0
            for j in range(n):
                tmp1 += self.a[j] * labels[j] * gamma[i][j]
            tmp = labels[i] * (tmp1 + self.b)
            if tmp <= 0:
                self.a[i] += self.step
                self.b = self.b + self.step * labels[i]
                step_count += 1
                correct_count = 0
            else:
                correct_count += 1
                if correct_count > features.shape[0]:
                    break
                else:
                    i += 1
                    i = i % n
        return (self.a, self.b)

    def predict(self, features):
        labels = features.dot(self.w) + self.b
        return labels




if __name__ == '__main__':
    features = np.array([[3, 3],
                   [4, 3],
                   [1, 1]])
    labels = np.array([1, 1, -1])

    perceptron = Perceptron()
    result = perceptron.train(features, labels)
    w, b = result

    predict_x = np.linspace(0, 5, 50)
    predict_y = (0 - b - w[0] * predict_x) / w[1]

    print(result)

    import sys
    sys.path.append('../helpers')
    import visualize
    visualize.plot(features, labels, predict_x, predict_y)

    result_dual = perceptron.train_dual(features, labels)
    print(result_dual)

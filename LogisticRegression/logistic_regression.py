import numpy as np
import math
import pandas as pd



class LogisticRegression:
    def __init__(self):
        self.step = 0.001
        self.max_step_count = 15000

    def train(self, features, labels):
        features_count = features.shape[0]
        features_d_count = features.shape[1]
        self.theta = np.zeros(features_d_count + 1)
        for i in range(self.max_step_count):
            i += 1
            for j in range(features_count):
                features_with_prefix = np.hstack([np.ones((features_count, 1)), features])
                # print('show')
                # print(self.theta)
                # print(features_with_prefix)
                # print(self.theta * features_with_prefix[j])
                if i == 100:
                    midne = self.theta * features_with_prefix[j]
                    midle2 = math.exp(-np.sum(self.theta * features_with_prefix[j]))
                    # fe = features_with_prefix[j]
                    # print(features_with_prefix[j])
                    # print(self.theta)
                    # print((midne))

                error = 1 / (1 + math.exp(-np.sum(self.theta * features_with_prefix[j]))) - labels[j]
                a = self.step / features_count * error
                b = features[j]
                c = self.theta

                self.theta -= self.step / features_count * error * features_with_prefix[j]
                j += 1

    def train_by_vector(self, features, labels):
        features_count = features.shape[0]
        features_d_count = features.shape[1]
        self.theta = np.zeros([features_d_count + 1, 1])
        features_with_const = np.hstack([np.ones([features_count, 1]), features])
        i = 0
        while i <= self.max_step_count:
            i += 1
            sigmod = 1 / (1 + np.exp( - features_with_const.dot(self.theta)))
            error = sigmod - labels
            print(sum(error))
            self.theta = self.theta - self.step / features_count * (features_with_const.T).dot(error)


    def predict(self, features):
        features_count = features.shape[0]
        prediction = np.zeros(features_count)
        for j in range(features_count):
            prediction[j] = 1 / float(1 + math.exp(np.sum(self.theta * features[j])))
            j += 1

        return prediction




if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    features = np.array(df.iloc[:, 0:2])
    labels = np.array(df.iloc[:, 2])

    predict_x = np.linspace(0, 100, 100)
    logistic_regression = LogisticRegression()
    # logistic_regression.train(features, labels)
    labels_ = labels.reshape([-1, 1])
    logistic_regression.train_by_vector(features, labels_)
    predict_y = (- logistic_regression.theta[0] - logistic_regression.theta[1] * predict_x) / logistic_regression.theta[2]
    print(logistic_regression.theta)

    import sys
    sys.path.append('../helpers')
    import visualize
    visualize.plot(features, labels, predict_x, predict_y)
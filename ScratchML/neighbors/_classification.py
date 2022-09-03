from logging import raiseExceptions
import numpy as np
from operator import itemgetter


class KNNClassifier:

    def __init__(self,
                 k=3,
                 weights="uniform",
                 algorithm="auto",
                 leaf_size=30,
                 p=2,
                 n_jobs=None):
        self.k = k
        if weights not in ["uniform","distance"]:
            raise Exception("Incorrect Parameter")
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs
        self.X = None
        self.y = None
        self.classes = None
        self._trained = False

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.classes = np.unique(self.y)
        self._trained = True

    def minokswhi_distance(self, Xq, Xi):
        self.distance = 0
        for index, cell_value in enumerate(Xi):
            if self.p > 1:
                self.distance += (Xq[index] - cell_value)**self.p
            else:
                self.distance += np.abs(Xq[index] - cell_value)
        return self.distance**(1 / self.p)

    def distance_calc(self, Xq, return_distance=True):
        self.result = []
        if return_distance:
            for idx, value in enumerate(self.X):
                self.distance = self.minokswhi_distance(Xq, value)
                self.result.append((idx, self.distance, self.y[idx]))
        return sorted(self.result, key=itemgetter(1))[:self.k]

    def predict(self, Xq):
        if self._trained == True:
            self.Xq = Xq
            if self.weights == "uniform":
                self.result = self.distance_calc(Xq)
                return np.argmax([x[2] for x in self.result], axis=0)
            elif self.weights == "distance":
                self.result = self.distance_calc(Xq)
                self.weighted_result = {}
                for _, dist, label in self.result:
                    if label not in self.weighted_result:
                        self.weighted_result[label] = (1 / (dist + 0.001))
                    else:
                        self.weighted_result[label] += (1 / (dist + 0.001))
                return max(self.weighted_result,
                           key=lambda x: self.weighted_result[x])
        else:
            raise Exception("Fit the model, before prediction")

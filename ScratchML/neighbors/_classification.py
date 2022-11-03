from logging import raiseExceptions
import numpy as np
from operator import itemgetter
from joblib import Parallel, delayed, cpu_count


class KNNClassifier:

    def __init__(self,
                 k:int =3,
                 weights: str ="uniform",
                 p: int = 2,
                 n_jobs:int =1):
        self.k = k
        if weights not in ["uniform","distance"]:
            raise Exception("Incorrect Parameter for 'weights'")
        self.weights = weights
        if n_jobs == 0 or n_jobs < -1:
            raise Exception("Incorrect Parameter for 'n_jobs'")
        self.n_jobs = n_jobs
        self.X = None
        self.y = None
        self.p = p
        self.classes = None
        self._trained = False
        self.feature_count = 0
        self.index = None 

    def fit(self, X: list, y:list):
        self.slice_no = self.n_jobs
        if self.n_jobs == -1:
            self.slice_no = cpu_count()
        self.index = np.array_split(np.arange(len(X)),self.slice_no)
        self.feature_count = len(X[0])
        self.X = np.array_split(np.array(X), self.slice_no)
        self.y = np.array(y)
        self.classes = np.unique(self.y)
        self._trained = True

    def minkowshi_distance(self, Xq:list, Xi:list) -> float:
        self.distance = 0
        for index, cell_value in enumerate(Xi):
            if self.p > 1:
                self.distance += (Xq[index] - cell_value)**self.p
            else:
                self.distance += np.abs(Xq[index] - cell_value)
        return self.distance**(1 / self.p)

    def parallel_distance(self, Xq:list, Xi:list, idx_val: list):
        self.result = []
        for idx, value in enumerate(Xi):
            self.distance = self.minkowshi_distance(Xq, value)
            self.result.append((idx_val[idx], self.distance, self.y[idx_val[idx]]))
        return self.result

    def distance_calc(self,Xq:list):
        self.result = []
        self.result = Parallel(n_jobs=self.n_jobs)(delayed(self.parallel_distance)(Xq,self.X[i],self.index[i]) for i in range(len(self.X)))
        self.result = np.concatenate(self.result)
        return sorted(self.result, key=itemgetter(1))[:self.k]

    def predict(self, Xq:list):
        if self._trained:
            self.Xq = np.array(Xq)
            if len(self.Xq) != self.feature_count:
                raise Exception('Mismatch no. of columns')
            if self.weights == "uniform":
                self.result = self.distance_calc(Xq)
                return np.bincount([x[2] for x in self.result]).argmax()
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

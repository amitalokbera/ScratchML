import numpy as np
from ._classification import KNNClassifier
from ..transformation import Normalize


class KNNRegressor(KNNClassifier):

    def __init__(self,
                 k=3,
                 weights="uniform",
                 p=2,
                 n_jobs=1):
        super().__init__(k, weights,p, n_jobs)
        if weights == "distance":
            self.normalize = Normalize()

    def fit(self, X: list, y: list):
        return super().fit(X, y)
    
    def minkowshi_distance(self, Xq: list, Xi: list) -> float:
        return super().minkowshi_distance(Xq, Xi)
    
    def distance_calc(self, Xq: list):
        return super().distance_calc(Xq)
    
    def predict(self, Xq:list):
        if self._trained == True:
            self.Xq = np.array(Xq)
            if len(self.Xq) != self.feature_count:
                raise Exception('Mismatch no. of columns')
            if self.weights == "uniform":
                self.result = self.distance_calc(Xq)
                return np.mean([x[2] for x in self.result])
            elif self.weights == "distance":
                self.result = self.distance_calc(Xq)
                self.weighted_result = {}
                for _, dist, label in self.result:
                    if label not in self.weighted_result:
                        self.weighted_result[label] = (1 / (dist + 0.001))
                    else:
                        self.weighted_result[label] += (1 / (dist + 0.001))
                self.normalize.fit(list(self.weighted_result.values()))
                self.result = 0
                for k,v in self.weighted_result.items():
                    self.val, self.distance_w = k,v
                    self.distance_w = self.normalize.transform([self.distance_w])[0]
                    self.result += self.val*self.distance_w
                return self.result
        else:
            raise Exception("Fit the model, before prediction") 
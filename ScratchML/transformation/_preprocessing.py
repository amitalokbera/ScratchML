import numpy as np


class StandardScaler:

    def __init__(self, copy: bool = True) -> None:
        self.copy = copy
        self.data = None
        self.mean = None
        self.std = None
        self.var = None
        self._fitted = False

    def fit(self, data: list):
        self.mean = np.mean(data)
        self.std = np.std(data)
        self.var = self.std**2
        self._fitted = True

    def transform(self, data: list):
        if self._fitted:
            if self.copy:
                self.data = data.copy()
            else:
                self.data = data
            self.data = [(x - self.mean) / self.std for x in self.data]
            return self.data
        else:
            raise Exception("Fit the transformation, before transform")


class Normalize:
    
    def __init__(self, copy: bool = True) -> None:
        self.copy = copy
        self.data = None
        self.min = None
        self.max = None
        self._fitted = False

    def fit(self, data: list):
        self.min = np.min(data)
        self.max = np.max(data)
        self._fitted = True

    def transform(self, data: list):
        if self._fitted:
            if self.copy:
                self.data = data.copy()
            else:
                self.data = data
            self.data = [(x - self.min) / (self.max - self.min) for x in self.data]
            return self.data
        else:
            raise Exception("Fit the transformation, before transform")
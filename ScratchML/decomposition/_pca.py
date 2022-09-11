import numpy as np

class PCA:
    def __init__(self, n_components:int = 2, random_state:int = 42) -> None:
        self.n_components = n_components
        self.random_state = 42
        np.random.seed(random_state)
        self.X = None

    def fit(self, X:np.ndarray):
        self.X = X 
import numpy as np
from joblib import Parallel, delayed


class LogisticRegression:
    def __init__(self,
                 penalty: str = "l2",
                 c: float = 1.0,
                 alpha:float = 0.01,
                 tol:float = 1e-4,
                 max_iter:int = 100,
                 batch_size:int = 100,
                 random_state: int = 42) -> None:
        self.penalty = penalty
        self.c = c 
        self.max_iter = max_iter 
        self.batch_size = batch_size
        np.random.seed(random_state)
        self.X = None 
        self.y = None 
        self._trained = False 
        self.coeff = None 
        self.classes = None 
        self.n_features = None 
        self.alpha = alpha
        self.tol = tol 

    def _initialize_weights(self, x_shape):
        self.coeff = np.random.sample(x_shape[1])
        self.n_features = x_shape[1]
        self.classes = np.unique(self.y)

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def fit(self,X, y):
        self.X, self.y = X, y
        self.X = np.hstack((np.ones((self.X.shape[0], 1)), self.X))
        self._initialize_weights(self.X.shape)
        self.costs = []
        for i in range(self.max_iter):
            z = self.X.dot(self.coeff)
            h = self.sigmoid(z)
            if self.penalty == 'l1':
                grad = self.X.T.dot(h - self.y) + self.c * np.sign(self.coeff)
            else:
                grad = self.X.T.dot(h - self.y) + 2*(self.c * self.coeff)
            self.coeff -= self.alpha * grad
            cost = -np.mean(self.y * np.log(h) + (1 - self.y) * np.log(1 - h)) + self.c * np.sum(np.abs(self.coeff))
            self.costs.append(cost)
            if i > 0 and np.abs(self.costs[-1] - self.costs[-2]) < self.tol:
                break
        
    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return self.sigmoid(X.dot(self.coeff)) > 0.5

    


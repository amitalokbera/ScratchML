import numpy as np
from joblib import Parallel, delayed
import cvxopt 
from ScratchML.linear_models._kernels import rbf_kernel, polynomial_kernel, linear_kernel
cvxopt.solvers.options['show_progress'] = False


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
        self._trained = True
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
        if self._trained is False:
            raise Exception('Fit the model first on training data')
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return self.sigmoid(X.dot(self.coeff)) > 0.5



class SVM(object):
    """The Support Vector Machine classifier.
    Uses cvxopt to solve the quadratic optimization problem.
    Parameters:
    -----------
    C: float
        Penalty term.
    kernel: function
        Kernel function. Can be either polynomial, rbf or linear.
    power: int
        The degree of the polynomial kernel. Will be ignored by the other
        kernel functions.
    gamma: float
        Used in the rbf kernel function.
    coef: float
        Bias term used in the polynomial kernel function.
    """
    def __init__(self, C=1, kernel=rbf_kernel, power=4, gamma=None, coef=4, random_state=42):
        self.C = C
        self.kernel = kernel
        self.power = power
        self.gamma = gamma
        self.coef = coef
        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.intercept = None
        np.random.seed(random_state)
        self._trained = False

    def fit(self, X, y):
        self._trained = True
        n_samples, n_features = np.shape(X)

        # Set gamma to 1/n_features by default
        if not self.gamma:
            self.gamma = 1 / n_features

        # Initialize kernel method with parameters
        self.kernel = self.kernel(
            power=self.power,
            gamma=self.gamma,
            coef=self.coef)

        # Calculate kernel matrix
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(X[i], X[j])

        # Define the quadratic optimization problem
        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

        if not self.C:
            G = cvxopt.matrix(np.identity(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G_max = np.identity(n_samples) * -1
            G_min = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h_max = cvxopt.matrix(np.zeros(n_samples))
            h_min = cvxopt.matrix(np.ones(n_samples) * self.C)
            h = cvxopt.matrix(np.vstack((h_max, h_min)))

        # Solve the quadratic optimization problem using cvxopt
        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        lagr_mult = np.ravel(minimization['x'])

        # Extract support vectors
        # Get indexes of non-zero lagr. multipiers
        idx = lagr_mult > 1e-7
        # Get the corresponding lagr. multipliers
        self.lagr_multipliers = lagr_mult[idx]
        # Get the samples that will act as support vectors
        self.support_vectors = X[idx]
        # Get the corresponding labels
        self.support_vector_labels = y[idx]

        # Calculate intercept with first support vector
        self.intercept = self.support_vector_labels[0]
        for i in range(len(self.lagr_multipliers)):
            self.intercept -= self.lagr_multipliers[i] * self.support_vector_labels[
                i] * self.kernel(self.support_vectors[i], self.support_vectors[0])

    def predict(self, X):
        if self._trained is False:
            raise Exception('Fit the model first on training data')
        y_pred = []
        # Iterate through list of samples and make predictions
        for sample in X:
            prediction = 0
            # Determine the label of the sample by the support vectors
            for i in range(len(self.lagr_multipliers)):
                prediction += self.lagr_multipliers[i] * self.support_vector_labels[
                    i] * self.kernel(self.support_vectors[i], sample)
            prediction += self.intercept
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)
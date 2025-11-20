import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class GradientModel:
    def __init__(self, degree: int, lr=0.01, n_iter=10000):
        self.degree = degree
        self.lr = lr
        self.n_iter = n_iter
        self.theta = None
        self.poly = PolynomialFeatures(degree=degree, include_bias=True)

    def fit(self, x_train, y_train):
        X = self.poly.fit_transform(x_train)
        m, n = X.shape
        self.theta = np.random.rand(n)

        for _ in range(self.n_iter):
            y_pred = X @ self.theta
            grad = (2 / m) * X.T @ (y_pred - y_train)
            self.theta -= self.lr * grad

    def predict(self, x):
        X = self.poly.transform(x)
        return X @ self.theta

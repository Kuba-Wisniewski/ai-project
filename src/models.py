import numpy as np
import torch


def relu(logits):
    return np.maximum(logits, 0)

def sigmoid(logits):
    return np.exp(-np.logaddexp(0, -logits))

def hardlim(logits):
    return np.round(sigmoid(logits))

def linear(logits):
    return logits


class HLayer:

    def __init__(self, f_act, n_inputs: int = 1, n_outputs: int = 1):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.W = np.random.normal(0, 1, (n_inputs, n_outputs))
        # to change?
        self.b = np.random.uniform(-np.pi, np.pi, size = (1, n_outputs))
        self.f_act = f_act

    def forward(self, X):
        return self.f_act(X @ self.W + self.b)


# weights init to 0 and no bias
class OLayer:

    def __init__(self, f_act, n_inputs: int = 1, n_outputs: int = 1):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.W = np.zeros((n_inputs, n_outputs))
        self.f_act = f_act

    def forward(self, X):
        return self.f_act(X @ self.W)


class NNModel:

    def __init__(self, f_act, n_inputs: int = 1, n_outputs: int = 1, n_hidden: int = 10, reg: int = 0):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.f_act = f_act

        self.hidden = HLayer(f_act, n_inputs, n_hidden)
        self.output = OLayer(linear, n_hidden, n_outputs)

        self.reg = reg


    def forward(self, X):
        hidden_result = self.hidden.forward(X)
        return self.output.forward(hidden_result)


    def train_output(self, x_train, y_train):
        hidden = self.hidden.forward(x_train)

        if self.reg == 0:
            hidden_pinv = np.linalg.pinv(hidden)
            self.output.W = hidden_pinv.dot(y_train)
        else:
            # We use linalg.solve to find the solution to (H'H + reg*I) * W2 = H'y,
            # equivalent to W2 = (H'H + reg*I)^(-1) * H'y
            self.output.W = np.linalg.solve(hidden.T @ hidden + reg * np.eye(n_hidden), hidden.T @ y_train)


def test(x_train, y_train, x_test, y_test, n_hidden = 10, reg = 0, num_models = 20):
    all_preds = []

    for _ in range(num_models):
        model = NNModel(relu, n_hidden=n_hidden, reg=reg)
        model.train_output(x_train, y_train)
        y_pred = model.forward(x_test)
        all_preds.append(y_pred)

    all_preds = np.array(all_preds)  # shape: (num_models, n_test_samples)

    avg_pred = np.mean(all_preds, axis=0)  # shape: (n_test_samples,)
    bias_squared = np.mean((avg_pred - y_test) ** 2)
    variance = np.mean(np.var(all_preds, axis=0))

    return bias_squared, variance



def sweep_test(x_train, y_train, x_test, y_test, n_hidden = 10, n_reps = 100, reg = 0.0):
    """
    Calculate the mean test error for fitting the second layer of the network for a defined number of repetitions.
    Notice that `init_scale` is always set to 0 in this case.
    Inputs:
    - x_train (np.ndarray): train input data.
    - y_train (np.ndarray): train target data.
    - x_test (np.ndarray): test input data.
    - y_test (np.ndarray): test target data.
    - n_hidden (int, default = 10): size of hidden layer.
    - n_reps (int, default = 100): number of resamples for data.
    - reg (float, default = 0): regularization constant.

    Outputs:
    - (float): mean error for train data.
    """
    return np.mean(np.array([test(x_train, y_train, x_test, y_test, n_hidden=n_hidden, reg = reg)[1] for _ in range(n_reps)]))
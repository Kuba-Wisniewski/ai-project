import numpy as np

def relu(logits):
    return np.maximum(logits, 0)

def sigmoid(logits):
    return np.exp(-np.logaddexp(0, -logits))

def hardlim(logits):
    return np.round(sigmoid(logits))

def linear(logits):
    return logits


# random weights and bias values
class HLayer:

    def __init__(self, f_act, n_inputs: int = 1, n_outputs: int = 1):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.W = np.random.normal(0, 1, (n_inputs, n_outputs))
        self.b = np.random.uniform(-10, 10, size = (1, n_outputs))
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

    def __init__(self, f_hid, f_out, n_inputs: int = 1, n_outputs: int = 1, n_hidden: int = 10):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.hidden = HLayer(f_hid, n_inputs, n_hidden)
        self.output = OLayer(f_out, n_hidden, n_outputs)


    def forward(self, X):
        hidden_result = self.hidden.forward(X)
        return self.output.forward(hidden_result)


    def train_output(self, x_train, y_train):
        # h @ W = y
        # W = y @ h^-1 - faster than gradient

        hidden = self.hidden.forward(x_train)

        hidden_pinv = np.linalg.pinv(hidden)
        self.output.W = hidden_pinv @ y_train


def test(x_train, y_train, x_test, y_test, n_hid = 10, num_models = 20):
    all_preds = []

    for _ in range(num_models):
        model = NNModel(f_hid=relu, f_out=linear, n_hidden=n_hid)
        model.train_output(x_train, y_train)
        y_pred = model.forward(x_test)
        all_preds.append(y_pred)

    all_preds = np.array(all_preds)

    avg_pred = np.mean(all_preds, axis=0)
    bias_squared = np.mean((avg_pred - y_test) ** 2)
    variance = np.mean(np.var(all_preds, axis=0))

    return bias_squared, variance, avg_pred

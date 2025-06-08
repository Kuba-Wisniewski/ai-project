import numpy as np


def get_data_sin(noise_dev: float = 0.0) -> (np.array, np.array):
    x_test = np.linspace(-np.pi, np.pi, 100)
    y_test = np.sin(x_test)

    x_train = np.linspace(-np.pi, np.pi, 10)
    y_train = np.sin(x_train) + noise_dev * np.random.normal(size=10)

    return x_train.reshape(10,1), y_train.reshape(10,1), x_test.reshape(100,1), y_test.reshape(100,1)


def get_data_poly(noise_dev: float = 0.0) -> (np.array, np.array):
    x_test = np.linspace(-10, 10, 100)
    y_test = x_test ** 3 + x_test ** 2 + x_test

    x_train = np.linspace(-10, 10, 10)
    y_train = x_train ** 3 + x_train ** 2 + x_train + noise_dev * np.random.normal(size=10)

    return x_train.reshape(10,1), y_train.reshape(10,1), x_test.reshape(100,1), y_test.reshape(100,1)

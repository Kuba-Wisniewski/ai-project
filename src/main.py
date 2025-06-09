from data import *
from models import *
from plots import *

import random
import matplotlib.pyplot as plt


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    set_random_seed(2137)
    hidden_layer_sizes = np.unique(np.round(np.logspace(0, 3, 20))).astype(int)

    x_train, y_train, x_test, y_test = get_data_poly(0.05)

    biases, variances = [], []

    for hid_size in hidden_layer_sizes:
        new_bias, new_var, y_pred = test(x_train, y_train, x_test, y_test, hid_size)

        biases.append(new_bias)
        variances.append(new_var)

        # plot_fit(hid_size, x_train, y_train, x_test, y_test, y_pred)

    total_err = np.array(biases) + np.array(variances)
    plot_bias_variance(hidden_layer_sizes, biases, variances, total_err)


if __name__ == '__main__':
    main()

from data import *
from models import *
from plots import *

import torch
import random
import matplotlib.pyplot as plt


def set_seed(seed = None, seed_torch = True):
    if seed is None:
        seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def main() -> None:
    set_seed(2137)
    n_hids = np.unique(np.round(np.logspace(0, 3, 20))).astype(int)

    x_train, y_train, x_test, y_test = get_data_sin(0.05)

    biases, variances = [], []

    for n_hid in n_hids:
        new_bias, new_var = test(x_train, y_train, x_test, y_test, n_hid, 0)

        biases.append(new_bias)
        variances.append(new_var)

    total_err = np.array(biases) + np.array(variances)
    plot_bias_variance(n_hids, biases, variances, total_err)


if __name__ == '__main__':
    main()

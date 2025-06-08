from data import *
from models import *

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

    x_train, y_train, x_test, y_test = get_data_poly(0.05)

    test_errs = [sweep_test(x_train, y_train, x_test, y_test, n_hidden=n_hid, n_reps=100, reg = 0.0) for n_hid in n_hids]

    plt.loglog(n_hids,test_errs,'o-',label='Test')
    plt.xlabel('Number of Hidden Units')
    plt.ylabel('Test Error')
    plt.show()


if __name__ == '__main__':
    main()

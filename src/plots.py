import matplotlib.pyplot as plt


def plot_fit(x_train, y_train, x_test, y_test, y_pred = 0):
    """
    Plot train and test data (as well as predicted values for test if given).

    Inputs:
    - x_train (np.ndarray): train input data.
    - y_train (np.ndarray): train target data.
    - x_test (np.ndarray): test input data.
    - y_test (np.ndarray): test target data.
    """
    plt.plot(x_test, y_test,label='Test data')
    plt.plot(x_train, y_train,'o',label='Training data')
    if y_pred:
        plt.plot(x_test, y_pred, label='Prediction')
    plt.legend()
    plt.xlabel('Input Feature')
    plt.ylabel('Target Output')
    plt.show()


def plot_predictions(n_hid, n_reps, x_train, y_train, x_test, y_test):
    """
    Generate train and test data for `n_reps` times, fit it for a network with hidden size `n_hid`, and plot prediction values.

    Inputs:
    - n_hid (int): size of hidden layer.
    - n_reps (int): number of data regenerations.
    """
    plt.plot(x_test, y_test,linewidth=4,label='Test data')
    plt.plot(x_train, y_train,'o',label='Training data')

    train_err, test_err, y_pred = fit_relu(x_train, y_train, x_test, y_test, n_hidden=n_hid)
    plt.plot(x_test, y_pred, color='g', label='Prediction')

    for rep in range(n_reps-1):
        train_err, test_err, y_pred = fit_relu(x_train, y_train, x_test, y_test, n_hidden=n_hid)
        plt.plot(x_test, y_pred, color='g', alpha=.5, label='_')

    plt.legend()
    plt.xlabel('Input Feature')
    plt.ylabel('Target Output')
    plt.title('Number of Hidden Units = {}'.format(n_hid))
    plt.show()
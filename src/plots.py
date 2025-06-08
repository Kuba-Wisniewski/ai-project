import matplotlib.pyplot as plt
import numpy as np

# not used - for testing
def plot_fit(x_train, y_train, x_test, y_test, y_pred = 0):
    plt.plot(x_test, y_test,label='Test data')
    plt.plot(x_train, y_train,'o',label='Training data')
    if y_pred:
        plt.plot(x_test, y_pred, label='Prediction')
    plt.legend()
    plt.xlabel('Input Feature')
    plt.ylabel('Target Output')
    plt.show()


def plot_bias_variance(n_hids, biases, variances, total_err) -> None:
    plt.figure(figsize=(10, 6))

    plt.loglog(n_hids, biases, 'o-', label='Bias^2', color='blue', linewidth=2)
    plt.loglog(n_hids, variances, 'o-', label='Variance', color='green', linewidth=2)
    plt.loglog(n_hids, total_err, 'o-', label='Total Error', color='red', linewidth=2.5, alpha=0.8)

    # interpolation threshold will have the maximum error value
    max_error_idx = np.argmax(total_err)
    interpolation_threshold_nhid = n_hids[max_error_idx]

    plt.axvline(x=interpolation_threshold_nhid, color='gray', linestyle='--', linewidth=1.5,
                label=f'Interpolation Threshold ({interpolation_threshold_nhid} hidden units)')


    plt.xlabel('Number of Hidden Units (Log Scale)')
    plt.ylabel('Error Value (Log Scale)')
    plt.title('Bias-Variance Tradeoff and Interpolation Threshold')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.savefig('fig2.png')
    plt.show()

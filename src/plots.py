import matplotlib.pyplot as plt
import numpy as np


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

    # Znajdź indeks, gdzie błąd całkowity jest MAKSYMALNY (często związany z progiem interpolacji)
    # W kontekście 'double descent', to jest punkt, w którym model właśnie
    # zaczął nadmiernie dopasowywać szum, a generalizacja jest najgorsza,
    # zanim wejdzie w reżim over-parametryzacji.
    max_error_idx = np.argmax(total_err)
    interpolation_threshold_nhid = n_hids[max_error_idx]

    # Dodaj pionową linię w miejscu progu interpolacji (maksymalnego błędu)
    plt.axvline(x=interpolation_threshold_nhid, color='gray', linestyle='--', linewidth=1.5,
                label=f'Interpolation Threshold ({interpolation_threshold_nhid} hidden units)')

    # Dodaj tekstową adnotację
    plt.text(interpolation_threshold_nhid * 1.1, np.max(total_err) * 0.9,
             f'Max Error\n@ {interpolation_threshold_nhid} H.U.',
             rotation=0, verticalalignment='top', fontsize=9, color='gray')


    plt.xlabel('Number of Hidden Units (Log Scale)')
    plt.ylabel('Error Value (Log Scale)')
    plt.title('Bias-Variance Tradeoff and Interpolation Threshold')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.show()
    plt.savefig('fig1.png')

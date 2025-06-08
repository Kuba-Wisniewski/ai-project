import matplotlib.pyplot as plt
import numpy as np


def plot_bias_variance_results(results: dict) -> None:
    degrees = list(results.keys())
    bias2 = [results[d]['bias2'] for d in degrees]
    variance = [results[d]['variance'] for d in degrees]
    total_error = [b + v for b, v in zip(bias2, variance)]

    plt.figure(figsize=(10, 6))
    plt.plot(degrees, bias2, label="Bias²", marker='o')
    plt.plot(degrees, variance, label="Variance", marker='s')
    plt.plot(degrees, total_error, label="Total Error", marker='^')
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Error")
    plt.title("Bias-Variance Trade-off")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_model_fits(degree: int, x_test: np.array, y_test: np.array, results: dict) -> None:
    plt.figure(figsize=(12, 8))

    for i, key in enumerate(results):
        result = results[key]
        y_pred = result['y_pred']
        x_train, y_train = result['x_train'], result['y_train']

        plt.plot(x_test, y_pred, label=f'Model {i + 1}', alpha=0.7)
        plt.scatter(x_train, y_train, s=20, color='black', zorder=5)

    plt.plot(x_test, y_test, color='red', linewidth=2, label='True function')
    plt.title(f'Regression Fits (Polynomial Degree = {degree})')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

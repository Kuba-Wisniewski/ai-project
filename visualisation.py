import numpy as np
import matplotlib.pyplot as plt
from data import *


# Wizualizacja wyników bias-variance
def plot_bias_variance_results(results: dict):
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


# Przykładowe wykresy dopasowania modelu
def plot_model_fits(degree: int, n_examples: int = 5):
    X_test = np.linspace(-1, 1, 200).reshape(-1, 1)
    y_true = np.sin(np.pi * X_test).ravel()

    plt.figure(figsize=(12, 8))
    for i in range(n_examples):
        X_train, y_train = generate_data()
        model = create_model(degree)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        plt.plot(X_test, y_pred, label=f'Model {i + 1}', alpha=0.7)
        plt.scatter(X_train, y_train, s=20, color='black', zorder=5)

    plt.plot(X_test, y_true, color='red', linewidth=2, label='True function')
    plt.title(f'Regression Fits (Polynomial Degree = {degree})')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

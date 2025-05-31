import numpy as np
from data import *


# Bias Variance Tradeoff analysis
def bias_variance_analysis(degrees: np.array, n_datasets: int = 100, n_test: int = 100) -> dict:
    x_test, y_test = generate_test_data(n_test)
    results = {}

    for degree in degrees:
        predictions = []

        for _ in range(n_datasets):
            x_train, y_train = generate_train_data()
            model = create_model(degree)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            predictions.append(y_pred)

        predictions = np.array(predictions)
        avg_pred = np.mean(predictions, axis=0)
        bias2 = np.mean((avg_pred - y_test) ** 2)
        variance = np.mean(np.var(predictions, axis=0))

        results[degree] = {'bias2': bias2, 'variance': variance}

    return results


def fitness_analysis(degree: int = 5, n_examples: int = 5, n_test: int = 200) -> (np.array, np.array, dict):
    x_test, y_test = generate_test_data(n_test)
    results = {}

    for i in range(n_examples):
        x_train, y_train = generate_train_data()
        model = create_model(degree)
        model.fit(x_train, y_train)  # change that to gradient descent
        y_pred = model.predict(x_test)

        results[i] = {'y_pred': y_pred, 'x_train': x_train, 'y_train': y_train}

    return x_test, y_test, results

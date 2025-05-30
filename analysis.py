import numpy as np
from data import *


# Analiza bias-variance
def bias_variance_analysis(degrees: np.array, n_datasets: int = 100, n_test: int = 100) -> dict:
    X_test = np.linspace(-1, 1, n_test).reshape(-1, 1)
    y_true = np.sin(np.pi * X_test).ravel()

    results = {}

    for degree in degrees:
        predictions = []

        for _ in range(n_datasets):
            X_train, y_train = generate_data()
            model = create_model(degree)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions.append(y_pred)

        predictions = np.array(predictions)
        avg_pred = np.mean(predictions, axis=0)
        bias2 = np.mean((avg_pred - y_true) ** 2)
        variance = np.mean(np.var(predictions, axis=0))
        results[degree] = {'bias2': bias2, 'variance': variance}

    return results

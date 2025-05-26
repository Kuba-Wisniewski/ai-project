import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# Funkcja do generowania danych
def generate_data(n_samples=30, noise_std=0.1):
    X = np.linspace(-1, 1, n_samples)
    y = np.sin(np.pi * X) + np.random.normal(0, noise_std, size=n_samples)
    return X.reshape(-1, 1), y


# Tworzenie modelu wielomianowego
def create_model(degree):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())


# Analiza bias-variance
def bias_variance_analysis(degrees, n_datasets=100, n_test=100):
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


# Wizualizacja wyników bias-variance
def plot_bias_variance_results(results):
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
def plot_model_fits(degree, n_examples=5):
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


# Uruchomienie analizy i wizualizacji
degrees = list(range(1, 21))
results = bias_variance_analysis(degrees)
plot_bias_variance_results(results)

# Przykładowe dopasowania modelu dla stopni 1, 5, 15
for deg in [1, 5, 15]:
    plot_model_fits(deg)
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline, Pipeline


# Funkcja do generowania danych
def generate_data(n_samples: int = 30, noise_std: int = 0.1) -> (np.array, np.array):
    X = np.linspace(-1, 1, n_samples)
    y = np.sin(np.pi * X) + np.random.normal(0, noise_std, size=n_samples)
    return X.reshape(-1, 1), y


# Tworzenie modelu wielomianowego
def create_model(degree: int) -> Pipeline:
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())

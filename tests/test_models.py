import src.data as dt
import src.models as md

import time
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sine_data():
    return dt.get_data_sin(0.05) 

@pytest.fixture
def trained_model(sine_data):
    x_train, y_train, _, _ = sine_data 
    model = md.NNModel(f_hid=md.relu, f_out=md.linear, n_hidden=100)
    model.train_output(x_train, y_train)
    return model


def test_accuracy(sine_data, trained_model) -> None:
    _, _, x_test, y_test = sine_data
    y_pred = trained_model.forward(x_test)

    max_err = 1e-1

    mse = np.mean((y_pred - y_test) ** 2)

    assert mse < max_err, (
        f"Model accuracy failed, MSE was {mse}"
    )


def test_latency(sine_data, trained_model) -> None:
    _, _, x_test, y_test = sine_data
    
    max_latency = 0.05

    start = time.perf_counter()
    trained_model.forward(x_test)
    end = time.perf_counter()
    duration = end - start

    assert duration < max_latency, (
        f"Model latency failed, duration was {duration}"
    )


def test_input_robustness(trained_model) -> None:
    inputs = np.array([[1000.0], [-1000.0], [0.5]])

    max_err = 10_000
    y_pred = trained_model.forward(inputs)

    assert not np.isnan(y_pred).any(), "Model robustness failed, returned NaN for extreme input"
    assert not np.isinf(y_pred).any(), "Model robustness failed, returned inf for extreme input"

    assert np.all((np.abs(y_pred) < max_err)), (
        f"Model robustness failed, max acceptable err was {max_err}"
    )


def test_output_bounds(sine_data, trained_model) -> None:
    _, _, x_test, _ = sine_data
    
    max_err = 1

    y_pred = trained_model.forward(x_test)
        
    assert np.min(y_pred) >= -1 - max_err, (
        f"Model output bounds failed, min val does not satisfy {max_err}"
    )

    assert np.max(y_pred) <= 1 + max_err, (
        f"Model output bounds failed, max val does not satisfy {max_err}"
    )
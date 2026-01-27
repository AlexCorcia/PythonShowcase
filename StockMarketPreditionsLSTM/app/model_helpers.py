import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_lstm_model_univariate(time_steps: int) -> tf.keras.Model:
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def plot_true_vs_pred(dates, y_true_1d, y_pred_1d, title: str):
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true_1d, label="True")
    plt.plot(dates, y_pred_1d, label="Predicted")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def compute_mae_rmse(y_true_1d: np.ndarray, y_pred_1d: np.ndarray):
    y_true_1d = np.asarray(y_true_1d).reshape(-1)
    y_pred_1d = np.asarray(y_pred_1d).reshape(-1)
    mae = float(np.mean(np.abs(y_true_1d - y_pred_1d)))
    rmse = float(np.sqrt(np.mean((y_true_1d - y_pred_1d) ** 2)))
    return mae, rmse

def build_lstm_model_multivariate(time_steps: int, n_features: int) -> tf.keras.Model:
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(time_steps, n_features)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)  # predicts return
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def returns_to_prices(start_price: float, returns_1d: np.ndarray) -> np.ndarray:
    returns_1d = np.asarray(returns_1d).reshape(-1)
    prices = np.empty_like(returns_1d, dtype=np.float64)
    p = float(start_price)
    for i, r in enumerate(returns_1d):
        p = p * (1.0 + float(r))
        prices[i] = p
    return prices


def plot_true_vs_pred_multi(dates, series: list[tuple[str, np.ndarray]], title: str, y_label: str = "Price"):
    plt.figure(figsize=(14, 6))
    for label, y in series:
        plt.plot(dates, np.asarray(y).reshape(-1), label=label)

    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

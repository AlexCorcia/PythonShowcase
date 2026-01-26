import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# ----------------------------
# Config
# ----------------------------
DEFAULT_SYMBOL = "MSFT"
DEFAULT_TIME_STEPS = 50
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 16

# Prefer env var: set ALPHAVANTAGE_API_KEY in your system
# but fallback to your current value so it runs.
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "S8HGL5NJILISUR66")


# ----------------------------
# Data helpers
# ----------------------------
def fetch_daily_data_alpha_vantage(symbol: str, api_key: str, outputsize: str = "compact") -> pd.DataFrame:
    """Fetch daily OHLCV data from Alpha Vantage and return a clean dataframe sorted by date asc."""
    ts = TimeSeries(key=api_key, output_format="pandas")
    data, _meta = ts.get_daily(symbol=symbol, outputsize=outputsize)

    data.columns = ["open", "high", "low", "close", "volume"]
    data = data.sort_index()  # oldest -> newest
    return data


def create_sequences(data_2d: np.ndarray, time_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    data_2d: shape (n, 1) scaled series
    returns:
      X: (n-time_steps, time_steps)
      y: (n-time_steps,)
    """
    X, y = [], []
    for i in range(time_steps, len(data_2d)):
        X.append(data_2d[i - time_steps:i, 0])
        y.append(data_2d[i, 0])
    return np.array(X), np.array(y)


def build_lstm_model(time_steps: int) -> tf.keras.Model:
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# ----------------------------
# Plot helpers
# ----------------------------
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


# ----------------------------
# Exercises (add new ones here)
# ----------------------------
def exercise_compact_80_20_fix_c(symbol: str = DEFAULT_SYMBOL,
                                time_steps: int = DEFAULT_TIME_STEPS,
                                epochs: int = DEFAULT_EPOCHS,
                                batch_size: int = DEFAULT_BATCH_SIZE):
    """
    Your current working exercise:
    - Alpha Vantage DAILY compact
    - 80/20 split by index
    - Fix C for test windows: last TIME_STEPS of train + all test
    - Train LSTM, predict test, plot with dates
    """
    if not API_KEY:
        raise ValueError("Missing Alpha Vantage API key. Set ALPHAVANTAGE_API_KEY env var.")

    # 1) Fetch
    data = fetch_daily_data_alpha_vantage(symbol=symbol, api_key=API_KEY, outputsize="compact")
    close_prices = data["close"].values

    # 2) Split
    train_size = int(len(close_prices) * 0.8)
    train_raw = close_prices[:train_size]
    test_raw = close_prices[train_size:]

    print("Total:", len(close_prices))
    print("Train:", len(train_raw), "Test:", len(test_raw))

    # 3) Scale (fit train only)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw.reshape(-1, 1))
    test_scaled = scaler.transform(test_raw.reshape(-1, 1))

    if len(train_scaled) < time_steps:
        raise ValueError(
            f"Not enough training data ({len(train_scaled)}) for TIME_STEPS={time_steps}. "
            "Reduce TIME_STEPS or fetch more data."
        )

    # 4) Sequences
    X_train, y_train = create_sequences(train_scaled, time_steps)
    X_train = X_train.reshape((X_train.shape[0], time_steps, 1))

    combined = np.vstack([train_scaled[-time_steps:], test_scaled])
    X_test, y_test = create_sequences(combined, time_steps)
    X_test = X_test.reshape((X_test.shape[0], time_steps, 1))

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test :", X_test.shape,  "y_test :", y_test.shape)

    # 5) Model
    model = build_lstm_model(time_steps=time_steps)
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # 6) Predict + inverse scale
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled).reshape(-1)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    print("Pred sample:", y_pred[:5])
    print("True sample:", y_true[:5])

    # 7) Plot
    test_dates = data.index[train_size:]  # 20 dates for compact with 80/20
    print("len(test_dates):", len(test_dates))
    print("len(y_true):", len(y_true))
    print("len(y_pred):", len(y_pred))

    plot_true_vs_pred(test_dates, y_true, y_pred, title="LSTM Prediction (Test Set)")


# Placeholder for your next exercise
def exercise_train_2024_test_2025(symbol: str = DEFAULT_SYMBOL,
                                 time_steps: int = DEFAULT_TIME_STEPS,
                                 epochs: int = DEFAULT_EPOCHS,
                                 batch_size: int = DEFAULT_BATCH_SIZE):
    """
    Next exercise you described:
    - Train on all of 2024
    - Predict/evaluate all of 2025
    Note: You will need more than 'compact' (100 rows). Use a longer history endpoint/source.
    """
    raise NotImplementedError("Exercise not implemented yet. We'll add it next.")


# ----------------------------
# Main runner
# ----------------------------
def main():
    # Pick which exercise to run by name:
    EXERCISE_TO_RUN = "compact_80_20_fix_c"
    # EXERCISE_TO_RUN = "train_2024_test_2025"

    symbol = DEFAULT_SYMBOL

    exercises = {
        "compact_80_20_fix_c": lambda: exercise_compact_80_20_fix_c(symbol=symbol),
        "train_2024_test_2025": lambda: exercise_train_2024_test_2025(symbol=symbol),
    }

    if EXERCISE_TO_RUN not in exercises:
        raise ValueError(f"Unknown exercise '{EXERCISE_TO_RUN}'. Options: {list(exercises.keys())}")

    exercises[EXERCISE_TO_RUN]()


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries


def fetch_daily_data_alpha_vantage(symbol: str, api_key: str, outputsize: str = "compact") -> pd.DataFrame:
    """Fetch daily OHLCV data from Alpha Vantage and return a clean dataframe sorted by date asc."""
    ts = TimeSeries(key=api_key, output_format="pandas")
    data, _meta = ts.get_daily(symbol=symbol, outputsize=outputsize)
    data.columns = ["open", "high", "low", "close", "volume"]
    return data.sort_index()


def fetch_weekly_data_alpha_vantage(symbol: str, api_key: str) -> pd.DataFrame:
    """Fetch weekly OHLCV data from Alpha Vantage and return a clean dataframe sorted by date asc."""
    ts = TimeSeries(key=api_key, output_format="pandas")
    data, _meta = ts.get_weekly(symbol=symbol)
    data.columns = ["open", "high", "low", "close", "volume"]
    return data.sort_index()


def create_sequences_univariate(data_2d: np.ndarray, time_steps: int):
    """
    Univariate sliding window.

    data_2d: shape (n, 1) scaled series
    Returns:
      X: (n-time_steps, time_steps)
      y: (n-time_steps,)
    """
    X, y = [], []
    for i in range(time_steps, len(data_2d)):
        X.append(data_2d[i - time_steps:i, 0])
        y.append(data_2d[i, 0])
    return np.array(X), np.array(y)

def make_supervised_from_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds weekly returns column:
      ret_t = (close_t / close_{t-1}) - 1
    Drops first NaN row after pct_change.
    """
    out = df.copy()
    out["ret"] = out["close"].pct_change()
    return out.dropna()


def create_sequences_multivariate(X_2d: np.ndarray, y_1d: np.ndarray, time_steps: int):
    """
    X_2d: (n, n_features)
    y_1d: (n,)
    Returns:
      X_seq: (n-time_steps, time_steps, n_features)
      y_seq: (n-time_steps,)
    """
    Xs, ys = [], []
    for i in range(time_steps, len(X_2d)):
        Xs.append(X_2d[i - time_steps:i, :])
        ys.append(y_1d[i])
    return np.array(Xs), np.array(ys)

import numpy as np
import pandas as pd


def create_sequences_univariate(data_2d: np.ndarray, time_steps: int):
    X, y = [], []
    for i in range(time_steps, len(data_2d)):
        X.append(data_2d[i - time_steps:i, 0])
        y.append(data_2d[i, 0])
    return np.array(X), np.array(y)


def make_supervised_from_weekly(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret"] = out["close"].pct_change()
    return out.dropna()


def create_sequences_multivariate(X_2d: np.ndarray, y_1d: np.ndarray, time_steps: int):
    Xs, ys = [], []
    for i in range(time_steps, len(X_2d)):
        Xs.append(X_2d[i - time_steps:i, :])
        ys.append(y_1d[i])
    return np.array(Xs), np.array(ys)

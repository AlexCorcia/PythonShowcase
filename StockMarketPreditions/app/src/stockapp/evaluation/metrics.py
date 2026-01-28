import numpy as np


def compute_mae_rmse(y_true_1d: np.ndarray, y_pred_1d: np.ndarray):
    y_true_1d = np.asarray(y_true_1d).reshape(-1)
    y_pred_1d = np.asarray(y_pred_1d).reshape(-1)
    mae = float(np.mean(np.abs(y_true_1d - y_pred_1d)))
    rmse = float(np.sqrt(np.mean((y_true_1d - y_pred_1d) ** 2)))
    return mae, rmse

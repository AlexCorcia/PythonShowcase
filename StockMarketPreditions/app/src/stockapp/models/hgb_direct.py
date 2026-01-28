import math
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


class HGBDirectMultiHorizon:
    """
    Gradient Boosting Direct Multi-Horizon model.

    Predice retornos log acumulados directamente para h = 1..H.
    """

    def __init__(
        self,
        horizon: int = 52,
        max_iter: int = 800,
        learning_rate: float = 0.03,
        max_depth: int = 6,
        random_state: int = 42,
    ):
        self.horizon = int(horizon)
        self.params = dict(
            loss="squared_error",
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )
        self.models: dict[int, HistGradientBoostingRegressor] = {}

    def fit(self, X: np.ndarray, log_close: np.ndarray):
        """
        X: shape (n_samples, n_features)
        log_close: shape (n_samples,)
        """
        X = np.asarray(X)
        log_close = np.asarray(log_close)

        n = len(log_close)
        if n <= self.horizon:
            raise ValueError("No hay suficientes muestras para el horizonte elegido.")

        # entrenamos un modelo por horizonte
        for h in range(1, self.horizon + 1):
            # y_h = log(C_{t+h}) - log(C_t)
            y = log_close[h:] - log_close[:-h]
            X_h = X[:-h]

            model = HistGradientBoostingRegressor(**self.params)
            model.fit(X_h, y)
            self.models[h] = model

        return self

    def predict_returns(self, X_origin: np.ndarray) -> np.ndarray:
        """
        Devuelve retornos log acumulados para h = 1..H
        """
        X_origin = np.asarray(X_origin)

        preds = []
        for h in range(1, self.horizon + 1):
            preds.append(float(self.models[h].predict(X_origin)[0]))
        return np.asarray(preds, dtype=np.float64)

    def predict_prices(self, X_origin: np.ndarray, start_price: float) -> np.ndarray:
        """
        Convierte retornos log acumulados a precios
        """
        log_returns = self.predict_returns(X_origin)
        return np.asarray([start_price * math.exp(r) for r in log_returns], dtype=np.float64)

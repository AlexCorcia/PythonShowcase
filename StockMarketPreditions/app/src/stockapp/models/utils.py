import numpy as np


def returns_to_prices(start_price: float, returns_1d: np.ndarray) -> np.ndarray:
    returns_1d = np.asarray(returns_1d).reshape(-1)
    prices = np.empty_like(returns_1d, dtype=np.float64)
    p = float(start_price)
    for i, r in enumerate(returns_1d):
        p = p * (1.0 + float(r))
        prices[i] = p
    return prices

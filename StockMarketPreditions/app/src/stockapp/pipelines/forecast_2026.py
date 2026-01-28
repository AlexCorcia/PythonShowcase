import numpy as np
import pandas as pd

from stockapp.data.providers.alphavantage_prices import fetch_weekly_data_alpha_vantage
from stockapp.evaluation.plots import plot_true_vs_pred_multi
from stockapp.models.hgb_direct import HGBDirectMultiHorizon


def run_forecast_2026_hgb_direct_weekly(
    symbol: str,
    api_key: str,
    forecast_weeks: int = 52,
    max_iter: int = 800,
):
    """
    Entrena con histórico hasta 2025 (incluido) y genera una previsión COMPLETA de 2026
    (semanal, direct multi-horizon), sin usar datos reales de 2026.

    Output: SOLO gráfico de forecast 2026.
    """

    # 1) Cargar datos semanales
    data = fetch_weekly_data_alpha_vantage(symbol=symbol, api_key=api_key).sort_index()

    # 2) Train SOLO <= 2025
    train_df = data.loc[: "2025-12-31"].copy()
    if len(train_df) < 200:
        raise ValueError("Muy pocos puntos semanales. Usa un símbolo con más histórico.")

    # 3) Feature engineering (basado SOLO en train_df)
    df = train_df.copy()
    df["log_close"] = np.log(df["close"].astype(float))
    df["log_ret_1"] = df["log_close"].diff(1)

    # lags retornos
    for lag in [1, 2, 3, 4, 8, 12, 16, 20]:
        df[f"lr_lag_{lag}"] = df["log_ret_1"].shift(lag)

    # rolling stats retornos
    for win in [4, 8, 12, 20, 26, 52]:
        df[f"lr_ma_{win}"] = df["log_ret_1"].rolling(win).mean()
        df[f"lr_std_{win}"] = df["log_ret_1"].rolling(win).std()

    # volumen: log + lags + rolling
    df["log_vol"] = np.log(df["volume"].astype(float).replace(0, np.nan))
    for lag in [1, 2, 4, 8]:
        df[f"lv_lag_{lag}"] = df["log_vol"].shift(lag)
    for win in [4, 8, 12, 20, 52]:
        df[f"lv_ma_{win}"] = df["log_vol"].rolling(win).mean()

    # momentum
    for k in [4, 8, 12, 26]:
        df[f"mom_{k}"] = (df["close"] / df["close"].shift(k)) - 1.0

    # limpiamos NaNs por ventanas/lags
    df_feat = df.dropna().copy()

    # columnas de features
    feature_cols = [c for c in df_feat.columns if c.startswith(("lr_", "lv_", "mom_"))]
    if not feature_cols:
        raise ValueError("No se generaron features (feature_cols vacío).")

    # 4) Entrenar modelo (ahora vive en models/)
    model = HGBDirectMultiHorizon(
        horizon=int(forecast_weeks),
        max_iter=int(max_iter),
        learning_rate=0.03,
        max_depth=6,
        random_state=42,
    )

    X = df_feat[feature_cols].values
    log_close = df_feat["log_close"].values
    model.fit(X, log_close)

    # 5) Forecast desde el último punto con features válidas (última semana de 2025)
    origin_date = df_feat.index[df_feat.index <= "2025-12-31"][-1]
    X_origin = df_feat.loc[[origin_date], feature_cols].values
    close0 = float(train_df.loc[origin_date, "close"])

    future_prices = model.predict_prices(X_origin=X_origin, start_price=close0)

    # 6) Fechas futuras (semanales)
    future_dates = [origin_date + pd.Timedelta(days=7 * (i + 1)) for i in range(int(forecast_weeks))]

    # 7) Plot SOLO forecast
    plot_true_vs_pred_multi(
        dates=future_dates,
        series=[("Forecast 2026 (trained on <=2025)", future_prices)],
        title=f"Forecast 2026 (HGB Direct Multi-Horizon, weekly) — {symbol}",
        y_label="Closing Price",
    )

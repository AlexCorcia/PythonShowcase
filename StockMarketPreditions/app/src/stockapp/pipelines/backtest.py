import numpy as np
import pandas as pd

from stockapp.data.providers.alphavantage_prices import fetch_weekly_data_alpha_vantage
from stockapp.evaluation.plots import plot_true_vs_pred_multi
from stockapp.models.hgb_direct import HGBDirectMultiHorizon


def run_backtest_hgb_direct_weekly(
    symbol: str,
    api_key: str,
    start_year: int = 2016,
    end_year: int = 2025,
    forecast_weeks: int = 52,
    max_iter: int = 800,
    plot_last_year: bool = True,
):
    """
    Backtest walk-forward por años (semanal) usando HGB Direct Multi-Horizon.

    Para cada año Y:
      - Train: datos <= (Y-1)-12-31
      - Origin: último punto del train con features válidas
      - Predict: siguientes forecast_weeks semanas (direct multi-horizon)
      - Eval: solo las semanas que caen dentro del año Y (y que existan en el dataset)
      - Baseline: precio constante = close del origin

    Imprime métricas por año y media final, y opcionalmente grafica el último año.
    """

    data_all = fetch_weekly_data_alpha_vantage(symbol=symbol, api_key=api_key).sort_index()

    results = []
    last_plot_payload = None

    for year in range(start_year, end_year + 1):
        train_end = f"{year-1}-12-31"
        test_start = f"{year}-01-01"
        test_end = f"{year}-12-31"

        train_df = data_all.loc[:train_end].copy()
        if len(train_df) < 250:
            print(f"[SKIP] {year}: pocos datos train ({len(train_df)})")
            continue

        # --- Feature engineering SOLO con train_df (evita leakage) ---
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

        df_feat = df.dropna().copy()
        feature_cols = [c for c in df_feat.columns if c.startswith(("lr_", "lv_", "mom_"))]

        if not feature_cols or len(df_feat) < 200:
            print(f"[SKIP] {year}: features insuficientes (df_feat={len(df_feat)})")
            continue

        # --- Entrenar modelo ---
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

        # --- Origin: última fila con features válidas ---
        origin_date = df_feat.index[-1]
        X_origin = df_feat.loc[[origin_date], feature_cols].values
        close0 = float(train_df.loc[origin_date, "close"])

        # --- Qué semanas futuras (reales) evaluamos ---
        idx = data_all.index
        pos = idx.get_indexer([origin_date])[0]
        future_idx = idx[pos + 1 : pos + 1 + int(forecast_weeks)]
        future_idx_in_year = [d for d in future_idx if (d >= pd.Timestamp(test_start) and d <= pd.Timestamp(test_end))]

        if len(future_idx_in_year) < 5:
            print(f"[SKIP] {year}: pocas semanas evaluables en el año")
            continue

        H = len(future_idx_in_year)

        # --- Predicción (precio) y recorte a H semanas evaluables ---
        y_pred_full = model.predict_prices(X_origin=X_origin, start_price=close0)
        y_pred = y_pred_full[:H]

        # --- Verdad y baseline ---
        y_true = data_all.loc[future_idx_in_year, "close"].astype(float).values
        y_base = np.full_like(y_true, fill_value=close0, dtype=np.float64)

        # --- Métricas ---
        mae = float(np.mean(np.abs(y_true - y_pred)))
        mae_base = float(np.mean(np.abs(y_true - y_base)))

        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        rmse_base = float(np.sqrt(np.mean((y_true - y_base) ** 2)))

        true_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)
        dir_acc = float(np.mean((np.sign(true_diff) == np.sign(pred_diff)).astype(float))) if len(true_diff) else float("nan")

        print(
            f"[{year}] weeks={H:>2}  "
            f"MAE={mae:.2f} (base {mae_base:.2f})  "
            f"RMSE={rmse:.2f} (base {rmse_base:.2f})  "
            f"DirAcc={dir_acc:.2%}"
        )

        results.append({
            "year": year,
            "weeks_eval": H,
            "MAE": mae,
            "MAE_base": mae_base,
            "RMSE": rmse,
            "RMSE_base": rmse_base,
            "DirAcc": dir_acc,
        })

        if plot_last_year and year == end_year:
            last_plot_payload = (future_idx_in_year, y_true, y_pred, y_base)

    if not results:
        print("No hay resultados evaluables.")
        return

    df_res = pd.DataFrame(results).sort_values("year")

    print("\n=== MEDIA (todas las ventanas evaluadas) ===")
    print(df_res[["MAE", "MAE_base", "RMSE", "RMSE_base", "DirAcc"]].mean(numeric_only=True))

    # Plot del último año
    if plot_last_year and last_plot_payload is not None:
        dts, y_true, y_pred, y_base = last_plot_payload
        plot_true_vs_pred_multi(
            dates=list(dts),
            series=[
                ("True", y_true),
                ("Pred (HGB Direct Multi-H)", y_pred),
                ("Baseline (last close)", y_base),
            ],
            title=f"Backtest {end_year} (weekly) — {symbol}",
            y_label="Closing Price",
        )

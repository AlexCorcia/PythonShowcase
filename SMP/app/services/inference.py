import numpy as np
import pandas as pd


def _make_feature_frame_from_window(window: np.ndarray, feature_names):
    window = np.asarray(window, dtype=float)
    df = pd.DataFrame({"close": window})

    # base rolling features
    df["return_1"] = df["close"].pct_change().fillna(0.0)
    df["return_5"] = df["close"].pct_change(5).fillna(0.0)
    df["ma_5"] = df["close"].rolling(5).mean().bfill()
    df["ma_10"] = df["close"].rolling(10).mean().bfill()
    df["vol_5"] = df["close"].pct_change().rolling(5).std().fillna(0.0)

    # lag features: for a window ending at t-1, lag_1 should be latest observed value
    latest_first = window[::-1]
    feature_values = {}

    for name in feature_names:
        if name.startswith("lag_"):
            try:
                lag_n = int(name.split("_")[1])
                feature_values[name] = latest_first[lag_n - 1] if lag_n <= len(latest_first) else latest_first[-1]
            except Exception:
                feature_values[name] = 0.0
        elif name in df.columns:
            feature_values[name] = float(df[name].iloc[-1])
        elif name == "close":
            feature_values[name] = float(window[-1])
        else:
            feature_values[name] = 0.0

    return pd.DataFrame([feature_values], columns=feature_names)


def _prepare_input(bundle, window: np.ndarray):
    feature_names = bundle.get("features", ["close"])
    model_type = bundle.get("type", "baseline")
    x_scaler = bundle.get("x_scaler")

    feat_df = _make_feature_frame_from_window(window, feature_names)
    x = feat_df.to_numpy(dtype=float)

    if x_scaler is not None:
        x = x_scaler.transform(x)

    if model_type in ["lstm", "gru", "rnn", "sequence"]:
        # sequence models would need a different feature builder
        return x.reshape(1, x.shape[0], x.shape[1])

    return x.reshape(1, -1)


def _inverse_target(bundle, yhat):
    y_scaler = bundle.get("y_scaler")
    yhat = float(np.array(yhat).reshape(-1)[0])

    if y_scaler is None:
        return yhat

    inv = y_scaler.inverse_transform(np.array([[yhat]]))
    return float(inv.reshape(-1)[0])


def _baseline_predict(window: np.ndarray) -> float:
    window = np.asarray(window, dtype=float)

    if len(window) < 2:
        return float(window[-1])

    returns = np.diff(window) / window[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    short_r = np.mean(returns[-5:]) if len(returns) >= 5 else np.mean(returns)
    med_r = np.mean(returns[-20:]) if len(returns) >= 20 else np.mean(returns)

    blended_r = 0.7 * short_r + 0.3 * med_r
    blended_r = float(np.clip(blended_r, -0.05, 0.05))

    last_price = float(window[-1])
    next_price = last_price * (1.0 + blended_r)
    return float(max(next_price, 0.01))


def _predict_one_step(bundle, window: np.ndarray) -> float:
    model_name = bundle["name"]
    model_type = bundle["type"]
    model = bundle["model"]

    if model_name == "baseline" or model_type == "baseline":
        return _baseline_predict(window)

    x = _prepare_input(bundle, window)

    try:
        pred = model.predict(x)
    except TypeError:
        pred = model.predict(x, verbose=0)

    return _inverse_target(bundle, pred)


def run_backtest(df: pd.DataFrame, model, model_name: str, horizon: int = 52):
    bundle = model
    lookback = bundle.get("lookback", 32)

    closes = df["close"].to_numpy(dtype=float)
    preds = np.full_like(closes, fill_value=np.nan, dtype=float)

    for i in range(lookback, len(closes)):
        window = closes[i - lookback:i]
        preds[i] = _predict_one_step(bundle, window)

    plot_df = df.copy()
    plot_df["pred"] = preds

    mask = ~np.isnan(plot_df["pred"].to_numpy())
    y_true = plot_df.loc[mask, "close"].to_numpy()
    y_pred = plot_df.loc[mask, "pred"].to_numpy()

    mae = float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else np.nan
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else np.nan

    y_true_ret = np.sign(np.diff(y_true))
    y_pred_ret = np.sign(np.diff(y_pred))
    diracc = float(np.mean(y_true_ret == y_pred_ret)) * 100 if len(y_true_ret) else np.nan

    return {
        "plot_df": plot_df,
        "forecast_start_date": None,
        "metrics": {"MAE": mae, "RMSE": rmse, "DirAcc (%)": diracc},
        "notes": f"lookback={lookback}, model_type={bundle.get('type')}",
    }


def run_forecast(df: pd.DataFrame, model, model_name: str, freq: str, future_months: int):
    bundle = model
    lookback = bundle.get("lookback", 32)

    history = df.copy()
    closes = history["close"].to_list()

    last_date = pd.to_datetime(history["date"].iloc[-1])

    if freq == "weekly":
        step = pd.Timedelta(days=7)
        horizon = max(1, int(round(future_months * 4.345)))
    else:
        step = pd.Timedelta(days=1)
        horizon = max(1, int(round(future_months * 30)))

    future_dates = []
    future_preds = []

    for k in range(horizon):
        window = np.array(closes[-lookback:], dtype=float)
        yhat = _predict_one_step(bundle, window)

        next_date = last_date + step if k == 0 else future_dates[-1] + step
        future_dates.append(next_date)
        future_preds.append(yhat)
        closes.append(yhat)

    future_df = pd.DataFrame({
        "date": future_dates,
        "close": np.nan,
        "pred": future_preds,
    })

    plot_df = history.copy()
    plot_df["pred"] = np.nan
    plot_df = pd.concat([plot_df, future_df], ignore_index=True)

    return {
        "plot_df": plot_df,
        "forecast_start_date": future_dates[0] if future_dates else None,
        "settings": {
            "freq": freq,
            "future_months": future_months,
            "horizon_steps": horizon,
            "lookback": lookback,
            "model_type": bundle.get("type"),
        },
        "metrics": {},
    }
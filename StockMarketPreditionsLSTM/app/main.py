import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import HistGradientBoostingRegressor

from app.data_helpers import (
    fetch_weekly_data_alpha_vantage,
    make_supervised_from_weekly,
    create_sequences_multivariate,
)

from app.model_helpers import (
    build_lstm_model_multivariate,
    returns_to_prices,
    plot_true_vs_pred_multi,
    compute_mae_rmse,
)



# ----------------------------
# Config
# ----------------------------
DEFAULT_SYMBOL = "NVDA"
DEFAULT_TIME_STEPS = 50
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 16

# Prefer env var: set ALPHAVANTAGE_API_KEY in your system
# but fallback to your current value so it runs.
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "S8HGL5NJILISUR66")


# ----------------------------
# Exercises
# ----------------------------
def exercise_compact_80_20_fix_c(symbol: str = DEFAULT_SYMBOL,
                                time_steps: int = DEFAULT_TIME_STEPS,
                                epochs: int = DEFAULT_EPOCHS,
                                batch_size: int = DEFAULT_BATCH_SIZE):
    """
    Exercise #1:
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
    X_train, y_train = create_sequences_univariate(train_scaled, time_steps)
    X_train = X_train.reshape((X_train.shape[0], time_steps, 1))

    combined = np.vstack([train_scaled[-time_steps:], test_scaled])
    X_test, y_test = create_sequences_univariate(combined, time_steps)
    X_test = X_test.reshape((X_test.shape[0], time_steps, 1))

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test :", X_test.shape,  "y_test :", y_test.shape)

    # 5) Model
    model = build_lstm_model_univariate(time_steps=time_steps)
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
    test_dates = data.index[train_size:]  # aligns with test_raw
    plot_true_vs_pred(test_dates, y_true, y_pred, title="LSTM Prediction (Test Set)")

def exercise_train_2024_test_2025(symbol: str = DEFAULT_SYMBOL,
                                  time_steps: int = 26,
                                  epochs: int = DEFAULT_EPOCHS,
                                  batch_size: int = DEFAULT_BATCH_SIZE):
    """
    Exercise #2:
    - WEEKLY data (free-tier friendly)
    - Train on 2024
    - Test on 2025
    - Fix C for year split: last `time_steps` of 2024 + all of 2025
    """
    if not API_KEY:
        raise ValueError("Missing Alpha Vantage API key. Set ALPHAVANTAGE_API_KEY env var.")

    # 1) Fetch weekly data
    data = fetch_weekly_data_alpha_vantage(symbol=symbol, api_key=API_KEY)

    # 2) Slice by year
    train_df = data.loc["2024-01-01":"2024-12-31"]
    test_df = data.loc["2025-01-01":"2025-12-31"]

    if len(train_df) == 0:
        raise ValueError("No 2024 data found. Check symbol or data availability.")
    if len(test_df) == 0:
        raise ValueError("No 2025 data found. Check symbol or data availability.")

    train_prices = train_df["close"].values
    test_prices = test_df["close"].values

    print(f"Weekly rows — 2024: {len(train_prices)} | 2025: {len(test_prices)}")

    # 3) Scale (fit only on 2024)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_prices.reshape(-1, 1))
    test_scaled = scaler.transform(test_prices.reshape(-1, 1))

    if len(train_scaled) < time_steps:
        raise ValueError(
            f"Not enough 2024 weekly points ({len(train_scaled)}) for TIME_STEPS={time_steps}. "
            "Reduce time_steps (try 12) or use more training years."
        )

    # 4) Sequences
    X_train, y_train = create_sequences_univariate(train_scaled, time_steps)
    X_train = X_train.reshape((X_train.shape[0], time_steps, 1))

    combined = np.vstack([train_scaled[-time_steps:], test_scaled])
    X_test, y_test = create_sequences_univariate(combined, time_steps)
    X_test = X_test.reshape((X_test.shape[0], time_steps, 1))

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test :", X_test.shape,  "y_test :", y_test.shape)

    # 5) Model
    model = build_lstm_model_univariate(time_steps=time_steps)
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

    mae, rmse = compute_mae_rmse(y_true, y_pred)
    print(f"2025 Weekly MAE:  {mae:.4f}")
    print(f"2025 Weekly RMSE: {rmse:.4f}")

    # 7) Plot
    dates_2025 = test_df.index[:len(y_true)]
    plot_true_vs_pred(dates_2025, y_true, y_pred, title="LSTM trained on 2024 (weekly) → predicted 2025 (weekly)")

def exercise_train_2020_2024_test_2025_price(symbol: str = DEFAULT_SYMBOL,
                                             time_steps: int = 12,
                                             epochs: int = 30,
                                             batch_size: int = DEFAULT_BATCH_SIZE):
    """
    Ejercicio FINAL (precio semanal):
    - Weekly data
    - Train: 2020–2024
    - Test: 2025
    - Features: close + volume
    - Target: weekly closing price
    - Fix-C logic for year split
    """
    if not API_KEY:
        raise ValueError("Missing Alpha Vantage API key.")

    # 1) Fetch weekly data
    data = fetch_weekly_data_alpha_vantage(symbol=symbol, api_key=API_KEY)

    train_df = data.loc["2020-01-01":"2024-12-31"].copy()
    test_df  = data.loc["2025-01-01":"2025-12-31"].copy()

    if len(train_df) < (time_steps + 20):
        raise ValueError("Not enough training data.")
    if len(test_df) < 5:
        raise ValueError("Not enough test data.")

    print(f"Weekly rows — train: {len(train_df)} | test: {len(test_df)}")

    # 2) Build FEATURES and TARGET
    feature_cols = ["close", "volume"]

    X_train_raw = train_df[feature_cols].values
    X_test_raw  = test_df[feature_cols].values

    y_train_raw = train_df["close"].values.reshape(-1, 1)
    y_test_raw  = test_df["close"].values.reshape(-1, 1)

    # 3) Scale (fit ONLY on train)
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))

    X_train = x_scaler.fit_transform(X_train_raw)
    X_test  = x_scaler.transform(X_test_raw)

    y_train = y_scaler.fit_transform(y_train_raw).reshape(-1)
    y_test  = y_scaler.transform(y_test_raw).reshape(-1)

    # 4) Sequences (Fix-C for year split)
    X_test_combined = np.vstack([X_train[-time_steps:], X_test])
    y_test_combined = np.concatenate([y_train[-time_steps:], y_test])

    X_train_seq, y_train_seq = create_sequences_multivariate(X_train, y_train, time_steps)
    X_test_seq, y_test_seq   = create_sequences_multivariate(X_test_combined, y_test_combined, time_steps)

    print("X_train_seq:", X_train_seq.shape, "y_train_seq:", y_train_seq.shape)
    print("X_test_seq :", X_test_seq.shape,  "y_test_seq :", y_test_seq.shape)

    # 5) Model
    n_features = X_train_seq.shape[2]
    model = build_lstm_model_multivariate(time_steps=time_steps, n_features=n_features)

    model.fit(
        X_train_seq, y_train_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_seq, y_test_seq),
        verbose=1
    )

    # 6) Predict + inverse scale
    y_pred_scaled = model.predict(X_test_seq)
    y_pred = y_scaler.inverse_transform(y_pred_scaled).reshape(-1)
    y_true = y_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).reshape(-1)

    # 7) Metrics
    mae, rmse = compute_mae_rmse(y_true, y_pred)
    print(f"2025 Weekly PRICE | MAE: {mae:.3f} | RMSE: {rmse:.3f}")

    # 8) Plot
    dates_2025 = test_df.index[:len(y_true)]

    plot_true_vs_pred_multi(
        dates_2025,
        series=[
            ("True 2025", y_true),
            ("LSTM predicted", y_pred),
        ],
        title="Train 2020–2024 (weekly, close+volume) → Predict 2025 (price)",
        y_label="Closing Price"
    )

def exercise_train_upto_2025_test_2026_price(symbol: str = DEFAULT_SYMBOL,
                                             time_steps: int = 12,
                                             epochs: int = 40,
                                             batch_size: int = DEFAULT_BATCH_SIZE):
    """
    Ejercicio 2026 (máxima info posible):
    - Weekly data
    - Train: desde el inicio hasta 2025-12-31
    - Test: 2026-01-01 hasta lo que haya disponible (puede ser parcial)
    - Features: close + volume
    - Target: weekly closing price
    - Fix-C logic: últimos time_steps del train + todo el test
    - Plot: True 2026 vs Predicted 2026 (+ baseline opcional)
    """
    if not API_KEY:
        raise ValueError("Missing Alpha Vantage API key.")

    # 1) Fetch weekly data
    data = fetch_weekly_data_alpha_vantage(symbol=symbol, api_key=API_KEY)

    train_df = data.loc[: "2025-12-31"].copy()
    test_df  = data.loc["2026-01-01":].copy()  # todo lo que tengas de 2026 (quizá parcial)

    if len(train_df) < (time_steps + 30):
        raise ValueError(f"Not enough training rows ({len(train_df)}). Reduce time_steps or check data.")
    if len(test_df) < 3:
        raise ValueError(
            "No (or too little) 2026 data found in your dataset yet. "
            "Alpha Vantage may not have returned 2026 weeks for this symbol."
        )

    print(f"Weekly rows — train(<=2025): {len(train_df)} | test(2026): {len(test_df)}")

    # 2) Features + target
    feature_cols = ["close", "volume"]

    X_train_raw = train_df[feature_cols].values
    X_test_raw  = test_df[feature_cols].values

    y_train_raw = train_df["close"].values.reshape(-1, 1)
    y_test_raw  = test_df["close"].values.reshape(-1, 1)

    # 3) Scale (fit only on train)
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))

    X_train = x_scaler.fit_transform(X_train_raw)
    X_test  = x_scaler.transform(X_test_raw)

    y_train = y_scaler.fit_transform(y_train_raw).reshape(-1)
    y_test  = y_scaler.transform(y_test_raw).reshape(-1)

    # 4) Sequences (Fix-C)
    X_test_combined = np.vstack([X_train[-time_steps:], X_test])
    y_test_combined = np.concatenate([y_train[-time_steps:], y_test])

    X_train_seq, y_train_seq = create_sequences_multivariate(X_train, y_train, time_steps)
    X_test_seq, y_test_seq   = create_sequences_multivariate(X_test_combined, y_test_combined, time_steps)

    print("X_train_seq:", X_train_seq.shape, "y_train_seq:", y_train_seq.shape)
    print("X_test_seq :", X_test_seq.shape,  "y_test_seq :", y_test_seq.shape)

    # 5) Model
    n_features = X_train_seq.shape[2]
    model = build_lstm_model_multivariate(time_steps=time_steps, n_features=n_features)

    model.fit(
        X_train_seq, y_train_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_seq, y_test_seq),
        verbose=1
    )

    # 6) Predict + inverse scale
    y_pred_scaled = model.predict(X_test_seq)
    y_pred = y_scaler.inverse_transform(y_pred_scaled).reshape(-1)
    y_true = y_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).reshape(-1)

    mae, rmse = compute_mae_rmse(y_true, y_pred)
    print(f"2026 Weekly PRICE | MAE: {mae:.3f} | RMSE: {rmse:.3f}")

    # 7) Baseline opcional (muy recomendable):
    # baseline = "la próxima semana será igual que esta semana"
    # Para alinearlo con y_true (targets), usamos los closes reales "previos" del test_combined.
    prev_close_scaled = y_test_combined[time_steps-1:-1]  # close de la semana anterior (escalado)
    baseline = y_scaler.inverse_transform(prev_close_scaled.reshape(-1, 1)).reshape(-1)

    mae_b, rmse_b = compute_mae_rmse(y_true, baseline)
    print(f"2026 Weekly BASE  | MAE: {mae_b:.3f} | RMSE: {rmse_b:.3f}")

    # 8) Plot
    dates_2026 = test_df.index[:len(y_true)]
    plot_true_vs_pred_multi(
        dates_2026,
        series=[
            ("True 2026", y_true),
            ("LSTM predicted", y_pred),
            ("Baseline (last close)", baseline),
        ],
        title="Train <=2025 (weekly, close+volume) → Predict 2026 (price) (" + DEFAULT_SYMBOL + ")",
        y_label="Closing Price"
    )

def exercise_pure_forecast_2026_weekly_price(symbol: str = DEFAULT_SYMBOL,
                                             time_steps: int = 12,
                                             epochs: int = 40,
                                             batch_size: int = DEFAULT_BATCH_SIZE,
                                             volume_strategy: str = "last",  # "last" o "mean"
                                             forecast_weeks: int = 52):
    """
    FORECAST PURO 2026 (sin usar datos reales de 2026 para comparar):
    - Weekly data (Alpha Vantage)
    - Train: todo hasta 2025-12-31 (máxima info posible)
    - Forecast: 52 semanas futuras (aprox. 1 año) a partir del último dato real disponible
    - Features: close + volume
    - Target: weekly closing price
    - Volumen futuro: asumido ('last' o 'mean')
    - Plot: SOLO la línea de predicción (sin True)
    """
    if not API_KEY:
        raise ValueError("Missing Alpha Vantage API key.")

    # 1) Fetch weekly data
    data = fetch_weekly_data_alpha_vantage(symbol=symbol, api_key=API_KEY)

    # Entrenamos con todo hasta fin de 2025
    train_df = data.loc[: "2025-12-31"].copy()
    if len(train_df) < (time_steps + 50):
        raise ValueError(
            f"Not enough training data rows ({len(train_df)}). "
            "Reduce time_steps or choose a symbol with more history."
        )

    print(f"Weekly rows — train(<=2025): {len(train_df)} | forecasting weeks: {forecast_weeks}")

    # 2) Features + target
    feature_cols = ["close", "volume"]

    X_train_raw = train_df[feature_cols].values
    y_train_raw = train_df["close"].values.reshape(-1, 1)

    # 3) Scale (fit only on train)
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))

    X_train = x_scaler.fit_transform(X_train_raw)
    y_train = y_scaler.fit_transform(y_train_raw).reshape(-1)

    # 4) Sequences para entrenar
    X_train_seq, y_train_seq = create_sequences_multivariate(X_train, y_train, time_steps)
    n_features = X_train_seq.shape[2]

    print("X_train_seq:", X_train_seq.shape, "y_train_seq:", y_train_seq.shape)

    # 5) Train model
    model = build_lstm_model_multivariate(time_steps=time_steps, n_features=n_features)
    model.fit(
        X_train_seq, y_train_seq,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # ----------------------------
    # FORECAST PURO (recursivo)
    # ----------------------------

    # Ventana inicial: últimas time_steps semanas reales disponibles (hasta 2025-12-31)
    start_window_df = train_df.iloc[-time_steps:].copy()

    # Estrategia para volumen futuro (porque no lo conocemos)
    if volume_strategy == "last":
        future_volume = float(start_window_df["volume"].iloc[-1])
    elif volume_strategy == "mean":
        future_volume = float(start_window_df["volume"].tail(time_steps).mean())
    else:
        raise ValueError("volume_strategy must be 'last' or 'mean'")

    # Ventana inicial escalada
    window_raw = start_window_df[feature_cols].values
    window_scaled = x_scaler.transform(window_raw)

    # Fechas futuras semanales: 52 semanas desde el último punto real del train
    last_date = train_df.index[-1]
    future_dates = [last_date + pd.Timedelta(days=7*(i+1)) for i in range(forecast_weeks)]

    future_preds = []

    for _ in range(forecast_weeks):
        # input shape: (1, time_steps, n_features)
        X_in = window_scaled.reshape(1, time_steps, n_features)

        # predicción del close (escalado)
        pred_close_scaled = model.predict(X_in, verbose=0)[0, 0]

        # inversa -> precio real
        pred_close = y_scaler.inverse_transform([[pred_close_scaled]])[0, 0]
        future_preds.append(pred_close)

        # Nuevo punto artificial: close predicho + volumen asumido
        new_point_raw = np.array([[pred_close, future_volume]], dtype=np.float64)
        new_point_scaled = x_scaler.transform(new_point_raw)

        # shift ventana
        window_scaled = np.vstack([window_scaled[1:], new_point_scaled])

    # ----------------------------
    # PLOT: SOLO FORECAST
    # ----------------------------
    plot_true_vs_pred_multi(
        future_dates,
        series=[
            ("Forecast 2026 (LSTM)", np.array(future_preds)),
        ],
        title=f"Pure Forecast 2026 (weekly, close+volume) — {symbol}",
        y_label="Closing Price"
    )

def exercise_hgb_forecast_2026_weekly(symbol: str = DEFAULT_SYMBOL,
                                      forecast_weeks: int = 52):
    """
    Modelo tabular (HistGradientBoostingRegressor) para predecir 2026 lo más preciso posible:
    - Weekly data
    - Train <= 2025
    - Eval 2026 disponible (si existe)
    - Forecast recursivo para completar 2026
    """

    if not API_KEY:
        raise ValueError("Missing Alpha Vantage API key.")

    data = fetch_weekly_data_alpha_vantage(symbol=symbol, api_key=API_KEY).copy()

    # --- train/test split por fecha ---
    train_df = data.loc[: "2025-12-31"].copy()
    test_2026_df = data.loc["2026-01-01":].copy()  # puede ser parcial

    if len(train_df) < 120:
        raise ValueError("Muy pocos puntos semanales. Usa un símbolo con más histórico o baja la complejidad.")

    # ---------- Feature engineering ----------
    def make_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        out["ret_1"] = out["close"].pct_change(1)
        out["ret_2"] = out["close"].pct_change(2)

        # lags de close
        for lag in [1, 2, 3, 4, 8, 12]:
            out[f"close_lag_{lag}"] = out["close"].shift(lag)

        # rolling stats
        for win in [4, 8, 12]:
            out[f"close_ma_{win}"] = out["close"].rolling(win).mean()
            out[f"close_std_{win}"] = out["close"].rolling(win).std()

        # volumen: lags + medias
        for lag in [1, 2, 4]:
            out[f"vol_lag_{lag}"] = out["volume"].shift(lag)

        for win in [4, 8, 12]:
            out[f"vol_ma_{win}"] = out["volume"].rolling(win).mean()

        # target: close de la semana siguiente
        out["y_next_close"] = out["close"].shift(-1)

        return out.dropna()

    feat_df = make_features(train_df)

    feature_cols = [c for c in feat_df.columns if c not in ["y_next_close"] and c not in ["open","high","low","close","volume"]]
    X_train = feat_df[feature_cols].values
    y_train = feat_df["y_next_close"].values

    # ---------- Model ----------
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=6,
        learning_rate=0.05,
        max_iter=800,
        random_state=42
    )
    model.fit(X_train, y_train)

    # ---------- Eval sobre 2026 disponible (opcional) ----------
    if len(test_2026_df) >= 20:
        eval_df = pd.concat([train_df.tail(30), test_2026_df], axis=0)
        eval_feat = make_features(eval_df)

        # quedarnos solo con filas cuya fecha esté en 2026
        eval_feat_2026 = eval_feat.loc["2026-01-01":]
        X_eval = eval_feat_2026[feature_cols].values
        y_true = eval_feat_2026["y_next_close"].values  # next close real
        y_pred = model.predict(X_eval)

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        print(f"[EVAL 2026 available] MAE: {mae:.3f} | RMSE: {rmse:.3f}")

        # plot eval (true vs pred) — solo en el rango disponible
        dates_eval = eval_feat_2026.index
        plot_true_vs_pred_multi(
            dates_eval,
            series=[("True next close (available)", y_true), ("Pred next close (available)", y_pred)],
            title=f"HGB Evaluate 2026 available (weekly) — {symbol}",
            y_label="Closing Price"
        )

    # ---------- Forecast puro 2026 (recursivo) ----------
    # arrancamos desde el último dato real de train (2025-12-31 o el último semanal disponible <=2025)
    hist = train_df.copy()

    future_dates = [hist.index[-1] + pd.Timedelta(days=7*(i+1)) for i in range(forecast_weeks)]
    future_preds = []

    # Para volumen futuro: mantenemos el último volumen conocido
    last_vol = float(hist["volume"].iloc[-1])

    # hacemos forecast semana a semana
    for _ in range(forecast_weeks):
        tmp = make_features(hist).tail(1)  # última fila con features válidas
        X_last = tmp[feature_cols].values
        next_close_pred = float(model.predict(X_last)[0])

        future_preds.append(next_close_pred)

        # añadimos “nueva semana” artificial con close predicho y volumen asumido
        new_row = pd.DataFrame(
            {"open": np.nan, "high": np.nan, "low": np.nan, "close": next_close_pred, "volume": last_vol},
            index=[hist.index[-1] + pd.Timedelta(days=7)]
        )
        hist = pd.concat([hist, new_row], axis=0)

    # Plot solo forecast
    plot_true_vs_pred_multi(
        future_dates,
        series=[("Forecast 2026 (HGB)", np.array(future_preds))],
        title=f"Pure Forecast 2026 (HGB, weekly features) — {symbol}",
        y_label="Closing Price"
    )


# ----------------------------
# Main runner
# ----------------------------
def main():
    # EXERCISE_TO_RUN = "train_2024_test_2025"
    # EXERCISE_TO_RUN = "compact_80_20_fix_c"
    # EXERCISE_TO_RUN = "train_2020_2024_test_2025_price"
    # EXERCISE_TO_RUN = "train_upto_2025_test_2026_price"
    # EXERCISE_TO_RUN = "exercise_pure_forecast_2026_weekly_price"
    EXERCISE_TO_RUN = "exercise_hgb_forecast_2026_weekly"



    symbol = DEFAULT_SYMBOL

    exercises = {
    "compact_80_20_fix_c": lambda: exercise_compact_80_20_fix_c(symbol=symbol),
    "train_2024_test_2025": lambda: exercise_train_2024_test_2025(symbol=symbol),
    "train_2020_2024_test_2025_price": lambda: exercise_train_2020_2024_test_2025_price(symbol=symbol),
    "train_upto_2025_test_2026_price": lambda: exercise_train_upto_2025_test_2026_price(symbol=symbol),
    "exercise_pure_forecast_2026_weekly_price": lambda: exercise_pure_forecast_2026_weekly_price(symbol=symbol),
    "exercise_hgb_forecast_2026_weekly": lambda: exercise_hgb_forecast_2026_weekly(symbol=symbol),

}

    if EXERCISE_TO_RUN not in exercises:
        raise ValueError(f"Unknown exercise '{EXERCISE_TO_RUN}'. Options: {list(exercises.keys())}")

    exercises[EXERCISE_TO_RUN]()


if __name__ == "__main__":
    main()

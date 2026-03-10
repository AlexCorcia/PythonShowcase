from pathlib import Path
import joblib
import pandas as pd
import lightgbm as lgb

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def make_lgbm_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    data = df.copy()
    data = data.sort_values("date").reset_index(drop=True)

    # lag features
    for i in range(1, lookback + 1):
        data[f"lag_{i}"] = data["close"].shift(i)

    # engineered features
    data["return_1"] = data["close"].pct_change()
    data["return_5"] = data["close"].pct_change(5)
    data["ma_5"] = data["close"].rolling(5).mean()
    data["ma_10"] = data["close"].rolling(10).mean()
    data["vol_5"] = data["close"].pct_change().rolling(5).std()

    # target = next close
    data["target"] = data["close"].shift(-1)

    data = data.dropna().reset_index(drop=True)
    return data


def train_lgbm_for_symbol(symbol: str, df: pd.DataFrame, lookback: int = 20):
    data = make_lgbm_features(df, lookback=lookback)

    feature_cols = [c for c in data.columns if c not in ["date", "close", "target"]]
    X = data[feature_cols]
    y = data["target"]

    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )

    model.fit(X, y)

    model_path = MODELS_DIR / f"lgbm_{symbol}.pkl"
    meta_path = MODELS_DIR / f"lgbm_{symbol}_meta.pkl"

    joblib.dump(model, model_path)

    meta = {
        "model_type": "lgbm",
        "lookback": lookback,
        "features": feature_cols,
        "x_scaler": None,
        "y_scaler": None,
    }
    joblib.dump(meta, meta_path)

    return {
        "symbol": symbol,
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "n_rows": len(data),
        "n_features": len(feature_cols),
    }
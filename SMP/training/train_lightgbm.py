import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

SYMBOL = "MSFT"

df = yf.download(SYMBOL, start="2015-01-01", auto_adjust=True)
df = df[["Close"]].rename(columns={"Close": "close"})

# ---------- features ----------

for i in range(1, 21):
    df[f"lag_{i}"] = df["close"].shift(i)

df["return_1"] = df["close"].pct_change()
df["return_5"] = df["close"].pct_change(5)

df["ma_5"] = df["close"].rolling(5).mean()
df["ma_10"] = df["close"].rolling(10).mean()

df["vol_5"] = df["close"].pct_change().rolling(5).std()

df["target"] = df["close"].shift(-1)

df = df.dropna()

features = [c for c in df.columns if c not in ["target", "close"]]

X = df[features]
y = df["target"]

# ---------- train model ----------

model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.02,
    max_depth=6,
)

model.fit(X, y)

# ---------- save model ----------

joblib.dump(model, f"models/lgbm_{SYMBOL}.pkl")

meta = {
    "model_type": "lgbm",
    "lookback": 20,
    "features": features,
}

joblib.dump(meta, f"models/lgbm_{SYMBOL}_meta.pkl")

print("Model saved.")
from pathlib import Path
import joblib

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def list_available_models(symbol: str, defaults=None):
    defaults = defaults or ["baseline"]

    files = list(MODELS_DIR.glob(f"*_{symbol}.*")) if MODELS_DIR.exists() else []
    names = []

    for f in files:
        base = f.stem
        if base.endswith("_meta"):
            continue
        if base.endswith(f"_{symbol}"):
            names.append(base.replace(f"_{symbol}", ""))
        else:
            names.append(base)

    merged = sorted(set(names) | set(defaults))
    return merged or ["baseline"]


def load_model(symbol: str, model_name: str):
    if model_name == "baseline":
        return {
            "name": "baseline",
            "type": "baseline",
            "model": None,
            "lookback": 32,
            "features": ["close"],
            "x_scaler": None,
            "y_scaler": None,
        }

    model_candidates = [
        MODELS_DIR / f"{model_name}_{symbol}.pkl",
        MODELS_DIR / f"{model_name}_{symbol}.joblib",
        MODELS_DIR / f"{model_name}_{symbol}.keras",
        MODELS_DIR / f"{model_name}_{symbol}.h5",
    ]

    model_path = None
    for path in model_candidates:
        if path.exists():
            model_path = path
            break

    if model_path is None:
        raise FileNotFoundError(
            f"Model not found for symbol={symbol!r} model_name={model_name!r}"
        )

    if model_path.suffix in [".pkl", ".joblib"]:
        model = joblib.load(model_path)
    elif model_path.suffix in [".keras", ".h5"]:
        from tensorflow.keras.models import load_model as keras_load_model
        model = keras_load_model(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_path.suffix}")

    meta_path = MODELS_DIR / f"{model_name}_{symbol}_meta.pkl"
    meta = {}
    if meta_path.exists():
        meta = joblib.load(meta_path)

    return {
        "name": model_name,
        "type": meta.get("model_type", model_name.lower()),
        "model": model,
        "lookback": meta.get("lookback", 32),
        "features": meta.get("features", ["close"]),
        "x_scaler": meta.get("x_scaler"),
        "y_scaler": meta.get("y_scaler"),
    }


def get_trained_models_inventory():
    inventory = []

    if not MODELS_DIR.exists():
        return inventory

    for f in sorted(MODELS_DIR.iterdir()):
        if not f.is_file():
            continue
        if f.suffix not in [".pkl", ".joblib", ".keras", ".h5"]:
            continue
        if f.stem.endswith("_meta"):
            continue

        stem = f.stem
        parts = stem.split("_")
        if len(parts) >= 2:
            model_name = "_".join(parts[:-1])
            symbol = parts[-1]
        else:
            model_name = stem
            symbol = "UNKNOWN"

        meta_path = MODELS_DIR / f"{model_name}_{symbol}_meta.pkl"
        has_meta = meta_path.exists()

        inventory.append({
            "symbol": symbol,
            "model_name": model_name,
            "file": f.name,
            "path": str(f),
            "has_meta": has_meta,
            "meta_file": meta_path.name if has_meta else None,
            "size_kb": round(f.stat().st_size / 1024, 2),
        })

    return inventory
"""
Utilidades compartidas para los experimentos de búsqueda de hiperparámetros
y comparación de modelos.

Centraliza:
- carga y limpieza de los datasets flat y temporal,
- definición de las columnas de features,
- preprocesado (scaler + one-hot),
- métricas y formato de guardado de resultados.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
CV_FOLDS = 5
PRIMARY_METRIC = "f1_macro"


FLAT_DATA_PATH = Path("app/data/processed/master_games_final.csv")
TEMPORAL_DATA_PATH = Path("app/data/processed/master_games_temporal.csv")

RESULTS_DIR = Path("app/results/hp_search")


FLAT_NUMERIC_FEATURES = [
    "num_moves",
    "player_captures",
    "player_checks",
    "aggression_score",
    "capture_rate",
    "check_rate",
]

FLAT_CATEGORICAL_FEATURES = ["eco_family"]

TEMPORAL_PHASE_FEATURES = [
    "opening_plies",
    "opening_player_captures",
    "opening_player_checks",
    "opening_player_castles",
    "opening_player_promotions",
    "opening_opponent_captures",
    "opening_opponent_checks",
    "middlegame_plies",
    "middlegame_player_captures",
    "middlegame_player_checks",
    "middlegame_player_castles",
    "middlegame_player_promotions",
    "middlegame_opponent_captures",
    "middlegame_opponent_checks",
    "endgame_plies",
    "endgame_player_captures",
    "endgame_player_checks",
    "endgame_player_castles",
    "endgame_player_promotions",
    "endgame_opponent_captures",
    "endgame_opponent_checks",
]

TEMPORAL_NUMERIC_FEATURES = ["num_moves"] + TEMPORAL_PHASE_FEATURES
TEMPORAL_CATEGORICAL_FEATURES = ["eco_family"]

TARGET = "style"


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    numeric_features: list[str]
    categorical_features: list[str]
    name: str
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)


def load_flat() -> Dataset:
    df = pd.read_csv(FLAT_DATA_PATH)
    df = df.dropna(subset=FLAT_NUMERIC_FEATURES + FLAT_CATEGORICAL_FEATURES + [TARGET])
    X = df[FLAT_NUMERIC_FEATURES + FLAT_CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()
    meta = df[["main_player"]].copy() if "main_player" in df.columns else pd.DataFrame()
    return Dataset(
        X=X,
        y=y,
        numeric_features=FLAT_NUMERIC_FEATURES,
        categorical_features=FLAT_CATEGORICAL_FEATURES,
        name="flat",
        meta=meta,
    )


def load_temporal() -> Dataset:
    df = pd.read_csv(TEMPORAL_DATA_PATH)
    df = df.dropna(
        subset=TEMPORAL_NUMERIC_FEATURES + TEMPORAL_CATEGORICAL_FEATURES + [TARGET]
    )
    X = df[TEMPORAL_NUMERIC_FEATURES + TEMPORAL_CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()
    meta = df[["main_player"]].copy() if "main_player" in df.columns else pd.DataFrame()
    return Dataset(
        X=X,
        y=y,
        numeric_features=TEMPORAL_NUMERIC_FEATURES,
        categorical_features=TEMPORAL_CATEGORICAL_FEATURES,
        name="temporal",
        meta=meta,
    )


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def evaluate_predictions(y_true, y_pred, labels=None) -> dict[str, Any]:
    if labels is None:
        labels = sorted(pd.Series(y_true).unique())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": {
            "labels": list(labels),
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_result(name: str, payload: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, ensure_ascii=False)
    return path


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self.t0

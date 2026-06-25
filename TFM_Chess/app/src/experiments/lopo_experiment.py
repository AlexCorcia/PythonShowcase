"""
Leave-one-player-out (LOPO) experiment.

Para cada jugador P y cada combinación (modelo, features):
- entrena con las partidas de los OTROS tres jugadores
- predice sobre las partidas de P
- registra la distribución de estilos predichos

Como el estilo de P es único en el dataset (positional ↔ Karpov, etc.), el modelo
nunca podrá predecir "el estilo correcto"; lo interesante es ver hacia qué
estilo de los tres entrenados se inclinan las partidas del jugador retirado.
Esto responde a la indicación del director:
  "Mirar si con otras métricas podemos ayudarnos a detectar el estilo de
   jugadores que no hemos entrenado."

Usa los mejores hiperparámetros encontrados en hp_search_classical.
Resultados en app/results/lopo/{features}.csv
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from app.src.experiments._common import (
    RANDOM_STATE,
    load_flat,
    load_temporal,
    make_preprocessor,
)

warnings.filterwarnings("ignore")


HP_DIR = Path("app/results/hp_search")
OUT_DIR = Path("app/results/lopo")


def load_best_params(model_name: str, features: str) -> dict:
    path = HP_DIR / f"{model_name}_{features}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["best_run"]["best_params"]


def strip_pipeline_prefix(params: dict) -> dict:
    return {k.split("__", 1)[1]: v for k, v in params.items() if k.startswith("model__")}


def build_estimator(model_name: str, params: dict, num_classes: int):
    bare = strip_pipeline_prefix(params)
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, **bare)
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **bare)
    if model_name == "xgboost":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **bare,
        )
    raise ValueError(model_name)


def run_lopo_for_dataset(dataset, models: list[str]) -> pd.DataFrame:
    if dataset.meta.empty:
        raise ValueError("dataset.meta no contiene main_player")
    X = dataset.X
    y = dataset.y
    main_players = dataset.meta["main_player"]

    players = sorted(main_players.unique().tolist())
    rows: list[dict] = []

    for model_name in models:
        try:
            best_params = load_best_params(model_name, dataset.name)
        except FileNotFoundError:
            print(
                f"  ! sin best params para {model_name}_{dataset.name}, lo salto"
            )
            continue

        for player in players:
            train_mask = main_players != player
            test_mask = main_players == player
            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]

            true_style = y_test.unique()[0] if len(y_test.unique()) == 1 else "MIXED"

            # XGBoost requiere labels enteros
            needs_le = model_name == "xgboost"
            if needs_le:
                le = LabelEncoder()
                y_train_enc = le.fit_transform(y_train)
                num_classes = len(le.classes_)
            else:
                y_train_enc = y_train
                num_classes = y_train.nunique()

            preprocessor = make_preprocessor(
                dataset.numeric_features, dataset.categorical_features
            )
            estimator = build_estimator(model_name, best_params, num_classes)
            pipe = Pipeline(
                steps=[("preprocessor", preprocessor), ("model", estimator)]
            )
            pipe.fit(X_train, y_train_enc)
            y_pred = pipe.predict(X_test)
            if needs_le:
                y_pred = le.inverse_transform(y_pred)

            distribution = pd.Series(y_pred).value_counts(normalize=True).to_dict()
            majority = max(distribution, key=distribution.get)

            for predicted_style, fraction in distribution.items():
                rows.append(
                    {
                        "features": dataset.name,
                        "model": model_name,
                        "held_out_player": player,
                        "true_style": true_style,
                        "predicted_style": predicted_style,
                        "fraction": round(fraction, 4),
                        "n_test": int(len(y_test)),
                        "majority_prediction": majority,
                    }
                )

            print(
                f"    {model_name:20s} hold-out={player:10s} "
                f"true={true_style:10s} majority={majority:10s} "
                f"n_test={len(y_test)}"
            )

    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = ["logistic_regression", "random_forest", "xgboost"]

    for loader, label in [(load_flat, "flat"), (load_temporal, "temporal")]:
        print(f"\n=== LOPO sobre features {label} ===")
        ds = loader()
        df = run_lopo_for_dataset(ds, models)
        out = OUT_DIR / f"{label}.csv"
        df.to_csv(out, index=False)
        print(f"  -> {out}")

    print("\nLOPO completo.")


if __name__ == "__main__":
    main()

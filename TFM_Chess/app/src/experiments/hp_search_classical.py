"""
Búsqueda sistemática de hiperparámetros para los modelos clásicos
sobre las representaciones flat y temporal.

Para cada combinación (modelo, features):
- split estratificado train/test (80/20)
- búsqueda de HP con StratifiedKFold(k=5), refit en F1-macro
- evaluación final sobre el test fijo
- guardado de best_params, CV score, métricas test, tiempo y tamaño del espacio

Resultados en app/results/hp_search/*.json
"""

from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from app.src.experiments._common import (
    CV_FOLDS,
    PRIMARY_METRIC,
    RANDOM_STATE,
    Dataset,
    Timer,
    evaluate_predictions,
    load_flat,
    load_temporal,
    make_preprocessor,
    save_result,
)


@dataclass
class ModelSpec:
    name: str
    estimator_factory: Callable[[int], Any]
    grid_space: dict[str, list]
    random_space: dict[str, list] | None
    needs_label_encoding: bool = False


def lr_factory(num_classes: int):
    return LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )


def rf_factory(num_classes: int):
    return RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def xgb_factory(num_classes: int):
    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


LR_SPEC = ModelSpec(
    name="logistic_regression",
    estimator_factory=lr_factory,
    grid_space={
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__class_weight": [None, "balanced"],
    },
    random_space=None,  # grid only per director note
)

RF_SPEC = ModelSpec(
    name="random_forest",
    estimator_factory=rf_factory,
    grid_space={
        "model__n_estimators": [200, 400],
        "model__max_depth": [10, 20, None],
        "model__min_samples_split": [2, 10],
        "model__min_samples_leaf": [1, 4],
    },
    random_space={
        "model__n_estimators": [100, 200, 300, 400, 600, 800],
        "model__max_depth": [6, 10, 14, 20, None],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", "log2", None],
        "model__class_weight": [None, "balanced"],
    },
)

XGB_SPEC = ModelSpec(
    name="xgboost",
    estimator_factory=xgb_factory,
    grid_space={
        "model__n_estimators": [300, 600],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.03, 0.1],
        "model__subsample": [0.8, 1.0],
    },
    random_space={
        "model__n_estimators": [200, 400, 600, 800, 1000],
        "model__max_depth": [3, 4, 5, 6, 7, 8],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "model__min_child_weight": [1, 3, 5, 7],
        "model__gamma": [0, 0.1, 0.3],
        "model__reg_alpha": [0, 0.1, 0.5, 1.0],
        "model__reg_lambda": [0.5, 1.0, 1.5, 2.0],
    },
    needs_label_encoding=True,
)


def run_search(
    spec: ModelSpec,
    dataset: Dataset,
    search_kind: str,
    n_iter_random: int = 40,
) -> dict[str, Any]:
    """Ejecuta una búsqueda (grid o random) y devuelve los resultados serializables."""

    y_for_split = dataset.y
    label_encoder = None
    if spec.needs_label_encoding:
        label_encoder = LabelEncoder()
        y_for_split = label_encoder.fit_transform(dataset.y)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        y_for_split,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_for_split,
    )

    num_classes = len(np.unique(y_for_split))
    preprocessor = make_preprocessor(
        dataset.numeric_features, dataset.categorical_features
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", spec.estimator_factory(num_classes)),
        ]
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    if search_kind == "grid":
        search = GridSearchCV(
            pipeline,
            param_grid=spec.grid_space,
            scoring=PRIMARY_METRIC,
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        space_size = int(np.prod([len(v) for v in spec.grid_space.values()]))
    elif search_kind == "random":
        if spec.random_space is None:
            raise ValueError(f"{spec.name} no tiene random_space definido")
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=spec.random_space,
            n_iter=n_iter_random,
            scoring=PRIMARY_METRIC,
            cv=cv,
            n_jobs=-1,
            refit=True,
            random_state=RANDOM_STATE,
            return_train_score=False,
        )
        space_size = n_iter_random
    else:
        raise ValueError(f"search_kind desconocido: {search_kind}")

    print(f"  [{spec.name} / {dataset.name} / {search_kind}] espacio={space_size}")
    with Timer() as t:
        search.fit(X_train, y_train)

    y_pred_test = search.predict(X_test)

    if label_encoder is not None:
        labels = list(label_encoder.classes_)
        y_test_lab = label_encoder.inverse_transform(y_test)
        y_pred_lab = label_encoder.inverse_transform(y_pred_test)
        test_metrics = evaluate_predictions(y_test_lab, y_pred_lab, labels=labels)
    else:
        labels = sorted(np.unique(y_for_split).tolist())
        test_metrics = evaluate_predictions(y_test, y_pred_test, labels=labels)

    return {
        "model": spec.name,
        "features": dataset.name,
        "search_kind": search_kind,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "search_space_size": space_size,
        "elapsed_seconds": round(t.elapsed, 2),
        "best_params": search.best_params_,
        "best_cv_f1_macro": float(search.best_score_),
        "test_metrics": test_metrics,
    }


def pick_best_result(*runs: dict[str, Any]) -> dict[str, Any]:
    """Para un mismo modelo+features, devuelve el run con mejor F1-macro CV."""
    return max(runs, key=lambda r: r["best_cv_f1_macro"])


PLAN = [
    # (spec, dataset_loader, [search_kinds])
    (LR_SPEC, load_flat, ["grid"]),
    (RF_SPEC, load_flat, ["grid", "random"]),
    (XGB_SPEC, load_flat, ["grid", "random"]),
    (RF_SPEC, load_temporal, ["random"]),
    (XGB_SPEC, load_temporal, ["random"]),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="filtrar por modelo (e.g. logistic_regression / random_forest / xgboost)",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        choices=[None, "flat", "temporal"],
        help="filtrar por tipo de features",
    )
    parser.add_argument(
        "--n-iter-random",
        type=int,
        default=40,
        help="número de iteraciones para RandomizedSearchCV",
    )
    args = parser.parse_args()

    # Cargamos los datasets una sola vez.
    flat = load_flat()
    temporal = load_temporal()
    loader_to_dataset = {load_flat: flat, load_temporal: temporal}

    for spec, loader, kinds in PLAN:
        if args.only and spec.name != args.only:
            continue
        dataset = loader_to_dataset[loader]
        if args.features and dataset.name != args.features:
            continue

        print(f"\n=== {spec.name} · {dataset.name} ===")
        runs = []
        for kind in kinds:
            run = run_search(spec, dataset, kind, n_iter_random=args.n_iter_random)
            runs.append(run)
            print(
                f"    {kind:6s}  CV f1_macro={run['best_cv_f1_macro']:.4f}  "
                f"test f1_macro={run['test_metrics']['f1_macro']:.4f}  "
                f"acc={run['test_metrics']['accuracy']:.4f}  "
                f"t={run['elapsed_seconds']}s"
            )

        best = pick_best_result(*runs)
        payload = {
            "model": spec.name,
            "features": dataset.name,
            "all_runs": runs,
            "best_run": best,
        }
        out_name = f"{spec.name}_{dataset.name}"
        path = save_result(out_name, payload)
        print(f"    -> {path}")


if __name__ == "__main__":
    main()

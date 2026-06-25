"""
Comparación en igualdad de condiciones (mismo split, mismas partidas de test).

Todos los modelos —clásicos y de grafos— se entrenan con EXACTAMENTE las mismas
partidas de entrenamiento (60%) y se evalúan sobre EXACTAMENTE las mismas
partidas de test (20%), emparejadas por una clave común
(main_player|white|black|date|result|eco) presente en ambos datasets.

Así la tabla resultante es una comparación pareada estricta: las diferencias en
F1-macro ya no pueden atribuirse a particiones distintas. Usa los mejores
hiperparámetros hallados en la búsqueda de cada modelo.

Salida: app/results/tables/unified_comparison.csv y .md
"""

from __future__ import annotations

import json
import warnings
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from torch_geometric.loader import DataLoader
from xgboost import XGBClassifier

from app.src.experiments._common import (
    FLAT_CATEGORICAL_FEATURES,
    FLAT_NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEMPORAL_CATEGORICAL_FEATURES,
    TEMPORAL_NUMERIC_FEATURES,
    evaluate_predictions,
    make_preprocessor,
)
from app.src.experiments.hp_search_gnn import (
    GCNClassifier,
    aggregate_to_games,
    predict_proba,
    train_one_epoch,
)
from app.src.experiments.train_temporal_gnn import (
    TemporalGNN,
    build_games,
    collate_games,
    evaluate as temporal_evaluate,
    iterate_batches,
)

warnings.filterwarnings("ignore")

HP_DIR = Path("app/results/hp_search")
OUT_DIR = Path("app/results/tables")
FIG_DIR = Path("app/results/figures")
GRAPH_JSON = Path("app/data/graphs/all_players_graphs.json")
GRAPH_PT = Path("app/data/graphs/graph_dataset.pt")
FLAT_CSV = Path("app/data/processed/master_games_final.csv")
TEMPORAL_CSV = Path("app/data/processed/master_games_temporal.csv")
STYLE_ORDER = ["defensive", "dynamic", "positional", "tactical"]


def slug(label: str) -> str:
    s = label.lower().replace("·", "").replace("+", "")
    for ch in " ()/áéíó":
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def plot_confusion(cm: dict, label: str, f1: float, acc: float, out: Path):
    arr = np.array(cm["matrix"], dtype=int)
    labels = cm["labels"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right"); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    ax.set_title(f"{label}\nF1-macro={f1:.3f} · acc={acc:.3f} (mismo test)")
    thr = arr.max() / 2 if arr.max() else 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, arr[i, j], ha="center", va="center",
                    color="white" if arr[i, j] > thr else "black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_unified_bars(df: pd.DataFrame, n_test: int, out: Path):
    d = df.sort_values("test_f1_macro")
    colors = ["#2e7d32" if "grafos" in m else "#5b8db8" for m in d["modelo"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(d))
    ax.barh(y, d["test_f1_macro"], color=colors)
    ax.set_yticks(y); ax.set_yticklabels(d["modelo"])
    ax.set_xlabel("F1-macro (test, mismo conjunto para todos)")
    ax.set_title(f"Comparativa en igualdad de condiciones\n(mismo entrenamiento y mismo test de {n_test} partidas)")
    ax.axvline(0.25, ls="--", color="gray", lw=1)
    for i, v in enumerate(d["test_f1_macro"]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, max(d["test_f1_macro"]) + 0.05)
    ax.legend(handles=[Patch(color="#2e7d32", label="Grafos"),
                       Patch(color="#5b8db8", label="Clásicos")], loc="lower right")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def make_key_series(df: pd.DataFrame) -> pd.Series:
    def norm(col):
        return df[col].astype(str).str.strip()
    return (
        norm("main_player") + "|" + norm("white") + "|" + norm("black") + "|"
        + norm("date") + "|" + norm("result") + "|" + norm("eco")
    )


def build_common_split():
    """Define la partición canónica 60/20/20 sobre las partidas comunes."""
    games = json.load(open(GRAPH_JSON, encoding="utf-8"))
    rows = []
    for g in games:
        m = g["metadata"]
        key = "|".join(
            str(m.get(k, "")).strip()
            for k in ["main_player", "white", "black", "date", "result", "eco"]
        )
        rows.append({"key": key, "style": m["style"], "game_index": m["game_index"]})
    gdf = pd.DataFrame(rows)

    # Partidas presentes también en el CSV clásico.
    flat = pd.read_csv(FLAT_CSV)
    flat["key"] = make_key_series(flat)
    common_keys = set(flat["key"]) & set(gdf["key"])
    gdf = gdf[gdf["key"].isin(common_keys)].drop_duplicates("key").reset_index(drop=True)

    keys = gdf["key"].tolist()
    labels = gdf["style"].tolist()
    k_trainval, k_test = train_test_split(
        keys, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    lab_tv = gdf.set_index("key").loc[k_trainval, "style"].tolist()
    k_train, k_val = train_test_split(
        k_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=lab_tv
    )
    split_of = {}
    for k in k_train:
        split_of[k] = "train"
    for k in k_val:
        split_of[k] = "val"
    for k in k_test:
        split_of[k] = "test"
    print(
        f"Partidas comunes={len(gdf)}  train={len(k_train)} val={len(k_val)} test={len(k_test)}"
    )
    return split_of


def load_best_params(name: str) -> dict:
    d = json.load(open(HP_DIR / f"{name}.json", encoding="utf-8"))
    return d["best_run"]["best_params"]


def bare_params(params: dict) -> dict:
    return {k.split("__", 1)[1]: v for k, v in params.items() if k.startswith("model__")}


def build_classical(model_name: str, params: dict, num_classes: int):
    p = bare_params(params)
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, **p)
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **p)
    if model_name == "xgboost":
        return XGBClassifier(
            objective="multi:softprob", num_class=num_classes, eval_metric="mlogloss",
            tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1, **p,
        )
    raise ValueError(model_name)


def eval_classical(label, model_name, csv_path, num_feats, cat_feats, json_name, split_of):
    df = pd.read_csv(csv_path)
    df["key"] = make_key_series(df)
    df["split"] = df["key"].map(split_of)
    df = df.dropna(subset=num_feats + cat_feats + [TARGET, "split"])
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    X_train, y_train = train[num_feats + cat_feats], train[TARGET]
    X_test, y_test = test[num_feats + cat_feats], test[TARGET]

    needs_le = model_name == "xgboost"
    if needs_le:
        le = LabelEncoder()
        y_tr = le.fit_transform(y_train)
        num_classes = len(le.classes_)
    else:
        y_tr = y_train
        num_classes = y_train.nunique()

    pipe = Pipeline([
        ("preprocessor", make_preprocessor(num_feats, cat_feats)),
        ("model", build_classical(model_name, load_best_params(json_name), num_classes)),
    ])
    pipe.fit(X_train, y_tr)
    y_pred = pipe.predict(X_test)
    if needs_le:
        y_pred = le.inverse_transform(y_pred)

    m = evaluate_predictions(y_test, y_pred, labels=STYLE_ORDER)
    return {"label": label, "n_test": int(len(y_test)), **m}


def graph_split_lists(dataset, split_of, key_of_index):
    out = {"train": [], "val": [], "test": []}
    for d in dataset:
        s = split_of.get(key_of_index.get(int(d.game_index)))
        if s is not None:
            out[s].append(d)
    return out


def class_weights(labels, num_classes, device):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    w = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(w, dtype=torch.float, device=device)


def eval_gnn_static(splits, device, seed=RANDOM_STATE, max_epochs=60, patience=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    params = load_best_params("gnn_graphs")
    input_dim = int(splits["train"][0].x.shape[1])
    num_classes = 4
    model = GCNClassifier(
        input_dim, params["hidden_dim"], num_classes,
        num_layers=params["num_layers"], dropout=params["dropout"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    cw = class_weights([int(d.y.item()) for d in splits["train"]], num_classes, device)

    train_loader = DataLoader(splits["train"], batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(splits["val"], batch_size=1024, shuffle=False)
    test_loader = DataLoader(splits["test"], batch_size=1024, shuffle=False)
    val_games = [int(d.game_index) for d in splits["val"]]
    val_labels = [int(d.y.item()) for d in splits["val"]]

    best_f1, best_state, no_imp = -1, None, 0
    for _ in range(max_epochs):
        train_one_epoch(model, train_loader, opt, device, cw)
        vp = predict_proba(model, val_loader, device)
        yt, yp = aggregate_to_games(vp, val_games, val_labels)
        f1 = evaluate_predictions(yt, yp, labels=STYLE_ORDER)["f1_macro"]
        if f1 > best_f1 + 1e-4:
            best_f1, best_state, no_imp = f1, deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    model.load_state_dict(best_state)
    tp = predict_proba(model, test_loader, device)
    yt, yp = aggregate_to_games(
        tp, [int(d.game_index) for d in splits["test"]], [int(d.y.item()) for d in splits["test"]]
    )
    m = evaluate_predictions(yt, yp, labels=STYLE_ORDER)
    return {"label": "GNN (estática + agregación) · grafos", "n_test": len(yt), **m}


def eval_gnn_temporal(dataset, split_of, key_of_index, device, seed=RANDOM_STATE, max_epochs=50, patience=10):
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    params = load_best_params("gnn_temporal_graphs")
    games = build_games(dataset)
    by_split = {"train": [], "val": [], "test": []}
    for gid, snaps in games.items():
        s = split_of.get(key_of_index.get(gid))
        if s is not None:
            by_split[s].append(snaps)
    input_dim = int(dataset[0].x.shape[1])
    num_classes = 4
    rng = random.Random(seed)

    model = TemporalGNN(
        input_dim, params["hidden_dim"], params["gru_hidden"], num_classes,
        num_layers=params["num_layers"], dropout=params["dropout"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=1e-5)
    cw = class_weights([int(s[0].y.item()) for s in by_split["train"]], num_classes, device)

    best_f1, best_state, no_imp = -1, None, 0
    for _ in range(max_epochs):
        model.train()
        for chunk in iterate_batches(by_split["train"], params["batch_size"], True, rng):
            big, owner, lengths, labels = collate_games(chunk, device)
            opt.zero_grad()
            logits = model(big, owner, lengths, device)
            loss = F.cross_entropy(logits, labels, weight=cw)
            loss.backward()
            opt.step()
        f1 = temporal_evaluate(model, by_split["val"], 256, device)["f1_macro"]
        if f1 > best_f1 + 1e-4:
            best_f1, best_state, no_imp = f1, deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    model.load_state_dict(best_state)
    m = temporal_evaluate(model, by_split["test"], 256, device)
    return {"label": "GNN temporal (GCN+GRU) · grafos", "n_test": len(by_split["test"]), **m}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    split_of = build_common_split()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    results = []

    print("\n== Modelos clásicos ==")
    for label, mname, csv, nf, cf, jname in [
        ("Regresión logística · flat", "logistic_regression", FLAT_CSV, FLAT_NUMERIC_FEATURES, FLAT_CATEGORICAL_FEATURES, "logistic_regression_flat"),
        ("Random Forest · flat", "random_forest", FLAT_CSV, FLAT_NUMERIC_FEATURES, FLAT_CATEGORICAL_FEATURES, "random_forest_flat"),
        ("XGBoost · flat", "xgboost", FLAT_CSV, FLAT_NUMERIC_FEATURES, FLAT_CATEGORICAL_FEATURES, "xgboost_flat"),
        ("Random Forest · temporal", "random_forest", TEMPORAL_CSV, TEMPORAL_NUMERIC_FEATURES, TEMPORAL_CATEGORICAL_FEATURES, "random_forest_temporal"),
        ("XGBoost · temporal", "xgboost", TEMPORAL_CSV, TEMPORAL_NUMERIC_FEATURES, TEMPORAL_CATEGORICAL_FEATURES, "xgboost_temporal"),
    ]:
        r = eval_classical(label, mname, csv, nf, cf, jname, split_of)
        results.append(r)
        print(f"  {label:32s} F1m={r['f1_macro']:.4f} acc={r['accuracy']:.4f} n_test={r['n_test']}")

    print("\n== Modelos de grafos ==")
    dataset = torch.load(GRAPH_PT, weights_only=False)
    games = json.load(open(GRAPH_JSON, encoding="utf-8"))
    key_of_index = {
        m["metadata"]["game_index"]: "|".join(
            str(m["metadata"].get(k, "")).strip()
            for k in ["main_player", "white", "black", "date", "result", "eco"]
        )
        for m in games
    }
    splits = graph_split_lists(dataset, split_of, key_of_index)
    print(f"  snapshots: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    # Las GNN se promedian sobre varias semillas (la inicialización de la red es
    # aleatoria): se reporta media ± desv. típica del F1-macro. Los modelos
    # clásicos son deterministas (random_state=42), una sola ejecución.
    SEEDS = [42, 7, 123, 2024, 99]

    for name, fn in [
        ("GNN (estática + agregación) · grafos", lambda s: eval_gnn_static(splits, device, seed=s)),
        ("GNN temporal (GCN+GRU) · grafos", lambda s: eval_gnn_temporal(dataset, split_of, key_of_index, device, seed=s)),
    ]:
        runs = [fn(s) for s in SEEDS]
        f1s = [r["f1_macro"] for r in runs]
        rep = runs[0]  # semilla 42 como representativa (matriz de confusión)
        rep["f1_macro_mean"] = float(np.mean(f1s))
        rep["f1_macro_std"] = float(np.std(f1s))
        rep["f1_macro_seeds"] = [round(x, 4) for x in f1s]
        rep["accuracy"] = float(np.mean([r["accuracy"] for r in runs]))
        rep["f1_weighted"] = float(np.mean([r["f1_weighted"] for r in runs]))
        results.append(rep)
        print(f"  {name:36s} F1m={rep['f1_macro_mean']:.4f}±{rep['f1_macro_std']:.4f} "
              f"seeds={rep['f1_macro_seeds']}")

    # Para clásicos, media=valor (deterministas) y std=0.
    for r in results:
        if "f1_macro_mean" not in r:
            r["f1_macro_mean"] = r["f1_macro"]
            r["f1_macro_std"] = 0.0

    df = pd.DataFrame([
        {"modelo": r["label"],
         "test_f1_macro": round(r["f1_macro_mean"], 4),
         "f1_macro_std": round(r["f1_macro_std"], 4),
         "test_accuracy": round(r["accuracy"], 4),
         "test_f1_weighted": round(r["f1_weighted"], 4),
         "n_test": r["n_test"]}
        for r in results
    ]).sort_values("test_f1_macro", ascending=False).reset_index(drop=True)

    csv_path = OUT_DIR / "unified_comparison.csv"
    md_path = OUT_DIR / "unified_comparison.md"
    df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Comparación en igualdad de condiciones (mismo split, mismas partidas de test)\n\n")
        f.write(f"Todas las filas comparten el MISMO conjunto de test ({df['n_test'].iloc[0]} partidas) ")
        f.write("y el mismo 60% de entrenamiento.\n\n")
        f.write(df.to_markdown(index=False))
    print(f"\n-> {csv_path}\n-> {md_path}")

    # Figuras en igualdad de condiciones: barras + matriz de confusión por modelo.
    n_test = int(df["n_test"].iloc[0])
    bars = FIG_DIR / "unified_comparison_f1_macro.png"
    plot_unified_bars(df, n_test, bars)
    print(f"-> {bars}")
    for r in results:
        out = FIG_DIR / f"cm_unified_{slug(r['label'])}.png"
        plot_confusion(r["confusion_matrix"], r["label"], r["f1_macro"], r["accuracy"], out)
        print(f"-> {out}")

    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()

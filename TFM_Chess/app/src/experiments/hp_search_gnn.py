"""
Búsqueda de hiperparámetros para la GNN sobre los grafos de posiciones.

Mejoras respecto a la versión inicial (ver app/docs/gnn_improvements_log.md):
- partición **por partida** (no por snapshot) -> sin fuga de información
- **agregación a nivel de partida**: la predicción de estilo de una partida es
  el promedio de las probabilidades de sus snapshots; esta es la métrica
  principal, comparable con los modelos clásicos
- **pesos de clase** en la función de pérdida
- random search en (hidden_dim, lr, weight_decay, dropout, num_layers, batch_size)

Resultado en app/results/hp_search/gnn_graphs.json
"""

from __future__ import annotations

import argparse
import random
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from app.src.experiments._common import (
    RANDOM_STATE,
    Timer,
    evaluate_predictions,
    save_result,
)

warnings.filterwarnings("ignore")


DATASET_PATH = Path("app/data/graphs/graph_dataset.pt")

LABEL_TO_STYLE = {0: "defensive", 1: "dynamic", 2: "positional", 3: "tactical"}
STYLE_ORDER = ["defensive", "dynamic", "positional", "tactical"]


class GCNClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3, dropout=0.3):
        super().__init__()
        assert num_layers >= 2
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.bns = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )
        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return x


def train_one_epoch(model, loader, optimizer, device, class_weight):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y.view(-1), weight=class_weight)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def predict_proba(model, loader, device):
    """Devuelve probabilidades softmax por snapshot, en el orden del loader."""
    model.eval()
    probs = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        probs.append(F.softmax(out, dim=1).cpu())
    return torch.cat(probs, dim=0) if probs else torch.empty(0)


def aggregate_to_games(probs, game_indices, snapshot_labels):
    """
    Agrega las probabilidades de los snapshots de cada partida (media) y
    devuelve, por partida: etiqueta real y predicción (estilo).
    """
    sum_probs: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(probs.shape[1]))
    counts: dict[int, int] = defaultdict(int)
    true_label: dict[int, int] = {}

    probs_np = probs.numpy()
    for i, gid in enumerate(game_indices):
        sum_probs[gid] += probs_np[i]
        counts[gid] += 1
        true_label[gid] = snapshot_labels[i]

    y_true, y_pred = [], []
    for gid in sorted(sum_probs.keys()):
        mean_prob = sum_probs[gid] / counts[gid]
        y_true.append(LABEL_TO_STYLE[true_label[gid]])
        y_pred.append(LABEL_TO_STYLE[int(mean_prob.argmax())])
    return y_true, y_pred


SEARCH_SPACE = {
    "hidden_dim": [64, 128, 256],
    "lr": [1e-4, 5e-4, 1e-3],
    "weight_decay": [1e-5, 1e-4, 1e-3],
    "dropout": [0.2, 0.3, 0.5],
    "num_layers": [2, 3],
    "batch_size": [256, 512],
}


def sample_config(rng):
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


def run_trial(config, splits, input_dim, num_classes, class_weight, max_epochs, patience, device):
    train_data, val_data, test_data = splits["train"], splits["val"], splits["test"]
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1024, shuffle=False)

    model = GCNClassifier(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        num_classes=num_classes,
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    val_games = [int(d.game_index) for d in val_data]
    val_labels = [int(d.y.item()) for d in val_data]

    best_val_f1 = -1.0
    best_state = None
    no_improve = 0
    train_losses = []

    for _ in range(1, max_epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, class_weight)
        train_losses.append(loss)

        val_probs = predict_proba(model, val_loader, device)
        yt, yp = aggregate_to_games(val_probs, val_games, val_labels)
        val_f1 = evaluate_predictions(yt, yp, labels=STYLE_ORDER)["f1_macro"]

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Métricas a nivel de partida (principal) y a nivel de snapshot (referencia).
    test_games = [int(d.game_index) for d in test_data]
    test_labels = [int(d.y.item()) for d in test_data]
    test_probs = predict_proba(model, test_loader, device)

    yt_game, yp_game = aggregate_to_games(test_probs, test_games, test_labels)
    game_metrics = evaluate_predictions(yt_game, yp_game, labels=STYLE_ORDER)

    snap_true = [LABEL_TO_STYLE[l] for l in test_labels]
    snap_pred = [LABEL_TO_STYLE[int(p.argmax())] for p in test_probs]
    snapshot_metrics = evaluate_predictions(snap_true, snap_pred, labels=STYLE_ORDER)

    return {
        "config": config,
        "epochs_trained": len(train_losses),
        "best_val_f1_macro": best_val_f1,
        "test_metrics": game_metrics,            # nivel partida (principal)
        "snapshot_test_metrics": snapshot_metrics,
        "n_test_games": len(yt_game),
    }


def stratified_game_split(dataset):
    """Particiona por partida (60/20/20) estratificando por estilo de la partida."""
    game_label: dict[int, int] = {}
    for d in dataset:
        game_label[int(d.game_index)] = int(d.y.item())

    games = sorted(game_label.keys())
    labels = [game_label[g] for g in games]

    g_trainval, g_test = train_test_split(
        games, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    lab_trainval = [game_label[g] for g in g_trainval]
    g_train, g_val = train_test_split(
        g_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=lab_trainval
    )

    train_set, val_set, test_set = set(g_train), set(g_val), set(g_test)
    splits = {"train": [], "val": [], "test": []}
    for d in dataset:
        gid = int(d.game_index)
        if gid in train_set:
            splits["train"].append(d)
        elif gid in val_set:
            splits["val"].append(d)
        else:
            splits["test"].append(d)
    return splits, len(g_train), len(g_val), len(g_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    rng = random.Random(RANDOM_STATE)

    print("Cargando dataset GNN...")
    dataset = torch.load(DATASET_PATH, weights_only=False)
    labels = [int(d.y.item()) for d in dataset]
    input_dim = int(dataset[0].x.shape[1])
    num_classes = len(set(labels))
    print(f"  snapshots={len(dataset)}, input_dim={input_dim}, classes={num_classes}")

    splits, n_gtr, n_gva, n_gte = stratified_game_split(dataset)
    print(
        f"  partidas: train={n_gtr} val={n_gva} test={n_gte} | "
        f"snapshots: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}")

    # Pesos de clase (inverso de frecuencia) calculados sobre el train.
    train_labels = np.array([int(d.y.item()) for d in splits["train"]])
    counts = np.bincount(train_labels, minlength=num_classes).astype(float)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    class_weight = torch.tensor(weights, dtype=torch.float, device=device)
    print(f"  class_weight={weights.round(3).tolist()}")

    sampled, seen = [], set()
    while len(sampled) < args.n_trials:
        c = sample_config(rng)
        key = tuple(sorted(c.items()))
        if key in seen:
            continue
        seen.add(key)
        sampled.append(c)

    trials = []
    for i, config in enumerate(sampled, start=1):
        print(f"\n[trial {i}/{args.n_trials}] {config}")
        with Timer() as t:
            res = run_trial(
                config, splits, input_dim, num_classes, class_weight,
                args.max_epochs, args.patience, device,
            )
        res["elapsed_seconds"] = round(t.elapsed, 2)
        print(
            f"  epochs={res['epochs_trained']} "
            f"val_game_f1={res['best_val_f1_macro']:.4f} "
            f"TEST game f1={res['test_metrics']['f1_macro']:.4f} "
            f"acc={res['test_metrics']['accuracy']:.4f} "
            f"(snapshot f1={res['snapshot_test_metrics']['f1_macro']:.4f}) "
            f"t={res['elapsed_seconds']}s"
        )
        trials.append(res)

    best = max(trials, key=lambda r: r["best_val_f1_macro"])

    payload = {
        "model": "gnn",
        "features": "graphs",
        "search_kind": "random",
        "evaluation_unit": "game (mean snapshot probabilities)",
        "n_train": len(splits["train"]),
        "n_test": len(splits["test"]),
        "n_train_games": n_gtr,
        "n_test_games": n_gte,
        "search_space": SEARCH_SPACE,
        "n_trials": args.n_trials,
        "all_trials": trials,
        "best_run": {
            "model": "gnn",
            "features": "graphs",
            "search_kind": "random",
            "n_train": n_gtr,
            "n_test": n_gte,
            "elapsed_seconds": best["elapsed_seconds"],
            "best_params": best["config"],
            "best_cv_f1_macro": best["best_val_f1_macro"],
            "test_metrics": best["test_metrics"],
            "snapshot_test_metrics": best["snapshot_test_metrics"],
        },
    }
    path = save_result("gnn_graphs", payload)
    print(f"\n=> guardado en {path}")
    print(
        f"BEST: val game f1={best['best_val_f1_macro']:.4f}  "
        f"test game f1={best['test_metrics']['f1_macro']:.4f}  "
        f"test game acc={best['test_metrics']['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()

"""
Búsqueda de hiperparámetros para la GNN sobre los grafos de posiciones.

Protocolo:
- split estratificado 60/20/20 (train/val/test) sobre los snapshots
- random search en el espacio (hidden_dim, lr, weight_decay, dropout, num_layers)
- cada configuración entrena hasta `max_epochs` con early stopping en val F1-macro
- se selecciona la configuración con mejor val F1-macro y se evalúa sobre test

Resultado en app/results/hp_search/gnn.json
"""

from __future__ import annotations

import argparse
import random
import warnings
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
    PRIMARY_METRIC,
    RANDOM_STATE,
    Timer,
    evaluate_predictions,
    save_result,
)

warnings.filterwarnings("ignore")


DATASET_PATH = Path("app/data/graphs/graph_dataset.pt")

LABEL_TO_STYLE = {
    0: "defensive",
    1: "dynamic",
    2: "positional",
    3: "tactical",
}


class GCNClassifier(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert num_layers >= 2

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return x


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=1)
        y_true.extend(batch.y.view(-1).cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
    return y_true, y_pred


SEARCH_SPACE = {
    "hidden_dim": [64, 128, 256],
    "lr": [1e-4, 5e-4, 1e-3],
    "weight_decay": [1e-5, 1e-4, 1e-3],
    "dropout": [0.2, 0.3, 0.5],
    "num_layers": [2, 3],
    "batch_size": [32, 64],
}


def sample_config(rng: random.Random) -> dict[str, Any]:
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


def run_trial(
    config: dict[str, Any],
    train_data,
    val_data,
    test_data,
    input_dim: int,
    num_classes: int,
    max_epochs: int,
    patience: int,
    device,
) -> dict[str, Any]:
    train_loader = DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    model = GCNClassifier(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        num_classes=num_classes,
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    best_val_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    train_losses = []

    for epoch in range(1, max_epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        train_losses.append(loss)

        y_true, y_pred = evaluate(model, val_loader, device)
        val_metrics = evaluate_predictions(
            y_true, y_pred, labels=list(range(num_classes))
        )
        val_f1 = val_metrics["f1_macro"]

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    # Cargar mejor estado y evaluar sobre test
    if best_state is not None:
        model.load_state_dict(best_state)
    y_true_test, y_pred_test = evaluate(model, test_loader, device)
    test_labels_str = [LABEL_TO_STYLE[i] for i in range(num_classes)]
    test_metrics = evaluate_predictions(
        [LABEL_TO_STYLE[i] for i in y_true_test],
        [LABEL_TO_STYLE[i] for i in y_pred_test],
        labels=test_labels_str,
    )

    y_true_val, y_pred_val = evaluate(model, val_loader, device)
    val_metrics = evaluate_predictions(
        [LABEL_TO_STYLE[i] for i in y_true_val],
        [LABEL_TO_STYLE[i] for i in y_pred_val],
        labels=test_labels_str,
    )

    return {
        "config": config,
        "epochs_trained": len(train_losses),
        "best_val_f1_macro": best_val_f1,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    rng = random.Random(RANDOM_STATE)

    print("Cargando dataset GNN...")
    dataset = torch.load(DATASET_PATH, weights_only=False)
    labels = [d.y.item() for d in dataset]
    input_dim = int(dataset[0].x.shape[1])
    num_classes = len(set(labels))
    print(f"  snapshots={len(dataset)}, input_dim={input_dim}, classes={num_classes}")

    # 60/20/20 estratificado
    idx_trainval, idx_test = train_test_split(
        list(range(len(dataset))),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    labels_trainval = [labels[i] for i in idx_trainval]
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=0.25,  # 0.25 * 0.8 = 0.2
        random_state=RANDOM_STATE,
        stratify=labels_trainval,
    )
    train_data = [dataset[i] for i in idx_train]
    val_data = [dataset[i] for i in idx_val]
    test_data = [dataset[i] for i in idx_test]
    print(f"  train={len(train_data)}  val={len(val_data)}  test={len(test_data)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}")

    sampled_configs: list[dict] = []
    seen = set()
    while len(sampled_configs) < args.n_trials:
        c = sample_config(rng)
        key = tuple(sorted(c.items()))
        if key in seen:
            continue
        seen.add(key)
        sampled_configs.append(c)

    trials: list[dict] = []
    for i, config in enumerate(sampled_configs, start=1):
        print(f"\n[trial {i}/{args.n_trials}] {config}")
        with Timer() as t:
            res = run_trial(
                config=config,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                input_dim=input_dim,
                num_classes=num_classes,
                max_epochs=args.max_epochs,
                patience=args.patience,
                device=device,
            )
        res["elapsed_seconds"] = round(t.elapsed, 2)
        print(
            f"  epochs={res['epochs_trained']}  "
            f"val f1_macro={res['best_val_f1_macro']:.4f}  "
            f"test f1_macro={res['test_metrics']['f1_macro']:.4f}  "
            f"acc={res['test_metrics']['accuracy']:.4f}  "
            f"t={res['elapsed_seconds']}s"
        )
        trials.append(res)

    best = max(trials, key=lambda r: r["best_val_f1_macro"])

    payload = {
        "model": "gnn",
        "features": "graphs",
        "search_kind": "random",
        "n_train": len(train_data),
        "n_val": len(val_data),
        "n_test": len(test_data),
        "search_space": SEARCH_SPACE,
        "n_trials": args.n_trials,
        "all_trials": trials,
        "best_run": {
            "model": "gnn",
            "features": "graphs",
            "search_kind": "random",
            "n_train": len(train_data),
            "n_test": len(test_data),
            "search_space_size": args.n_trials,
            "elapsed_seconds": best["elapsed_seconds"],
            "best_params": best["config"],
            "best_cv_f1_macro": best["best_val_f1_macro"],
            "test_metrics": best["test_metrics"],
        },
    }
    path = save_result("gnn_graphs", payload)
    print(f"\n=> guardado en {path}")
    print(
        f"BEST: val f1_macro={best['best_val_f1_macro']:.4f}  "
        f"test f1_macro={best['test_metrics']['f1_macro']:.4f}  "
        f"test acc={best['test_metrics']['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()

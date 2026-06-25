"""
Modelo temporal sobre grafos de posiciones (Tier 3).

A diferencia de la GNN estática (que clasifica snapshots de forma independiente),
este modelo trata cada partida como una **secuencia ordenada de snapshots**:

    snapshot_t --[GCN encoder + pooling]--> embedding_t
    (embedding_1, ..., embedding_T) --[GRU]--> estado final --> estilo de la partida

Así el modelo aprovecha la dimensión temporal del dato (la evolución de la
posición a lo largo de la partida), coherente con el título del TFM
"grafos temporales dinámicos". La predicción es directamente a nivel de partida.

- partición por partida (60/20/20), estratificada por estilo
- pesos de clase en la pérdida
- early stopping sobre F1-macro de validación (nivel partida)
- pequeña búsqueda aleatoria de hiperparámetros

Resultado en app/results/hp_search/gnn_temporal_graphs.json
"""

from __future__ import annotations

import argparse
import random
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence
from torch_geometric.data import Batch
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


class TemporalGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, gru_hidden, num_classes, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.bns = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.gru = torch.nn.GRU(hidden_dim, gru_hidden, batch_first=True)
        self.lin = torch.nn.Linear(gru_hidden, num_classes)

    def encode_snapshots(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        return global_mean_pool(x, batch)  # [num_snapshots, hidden_dim]

    def forward(self, big_batch, owner, lengths, device):
        emb = self.encode_snapshots(big_batch.x, big_batch.edge_index, big_batch.batch)
        B = lengths.size(0)
        Tmax = int(lengths.max().item())
        H = emb.size(1)
        N = emb.size(0)

        # Posición temporal de cada snapshot dentro de su partida, vectorizada.
        # `owner` agrupa los snapshots de cada partida de forma consecutiva, así
        # que basta restar el desplazamiento de inicio de la partida.
        lengths_dev = lengths.to(device)
        offsets = torch.zeros(B, dtype=torch.long, device=device)
        if B > 1:
            offsets[1:] = torch.cumsum(lengths_dev, dim=0)[:-1]
        time_idx = torch.arange(N, device=device) - offsets[owner]

        seq = torch.zeros(B, Tmax, H, device=device)
        seq[owner, time_idx] = emb

        packed = pack_padded_sequence(
            seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        return self.lin(h_n[-1])  # [B, num_classes]


def build_games(dataset):
    """game_index -> lista de snapshots ordenada por ply."""
    by_game = defaultdict(list)
    for d in dataset:
        by_game[int(d.game_index)].append(d)
    games = {}
    for gid, snaps in by_game.items():
        snaps_sorted = sorted(snaps, key=lambda d: int(d.ply_number))
        games[gid] = snaps_sorted
    return games


def collate_games(game_list, device):
    all_snaps, owner, lengths, labels = [], [], [], []
    for gi, snaps in enumerate(game_list):
        for d in snaps:
            all_snaps.append(d)
            owner.append(gi)
        lengths.append(len(snaps))
        labels.append(int(snaps[0].y.item()))
    big = Batch.from_data_list(all_snaps).to(device)
    return (
        big,
        torch.tensor(owner, dtype=torch.long, device=device),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long, device=device),
    )


def iterate_batches(game_items, batch_size, shuffle, rng):
    order = list(range(len(game_items)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        yield [game_items[j] for j in idx]


def evaluate(model, game_items, batch_size, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for chunk in iterate_batches(game_items, batch_size, False, None):
            big, owner, lengths, labels = collate_games(chunk, device)
            logits = model(big, owner, lengths, device)
            pred = logits.argmax(dim=1).cpu().tolist()
            y_true.extend(LABEL_TO_STYLE[l] for l in labels.cpu().tolist())
            y_pred.extend(LABEL_TO_STYLE[p] for p in pred)
    return evaluate_predictions(y_true, y_pred, labels=STYLE_ORDER)


SEARCH_SPACE = {
    "hidden_dim": [64, 128],
    "gru_hidden": [64, 128],
    "lr": [5e-4, 1e-3],
    "dropout": [0.2, 0.3],
    "num_layers": [2, 3],
    "batch_size": [128],
}


def stratified_game_split(games):
    gids = sorted(games.keys())
    labels = [int(games[g][0].y.item()) for g in gids]
    g_trainval, g_test = train_test_split(
        gids, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    lab_tv = [int(games[g][0].y.item()) for g in g_trainval]
    g_train, g_val = train_test_split(
        g_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=lab_tv
    )
    return (
        [games[g] for g in g_train],
        [games[g] for g in g_val],
        [games[g] for g in g_test],
    )


def run_trial(config, train_g, val_g, test_g, input_dim, num_classes, class_weight, max_epochs, patience, device, rng):
    model = TemporalGNN(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        gru_hidden=config["gru_hidden"],
        num_classes=num_classes,
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-5)

    best_val_f1, best_state, no_improve = -1.0, None, 0
    for _ in range(1, max_epochs + 1):
        model.train()
        for chunk in iterate_batches(train_g, config["batch_size"], True, rng):
            big, owner, lengths, labels = collate_games(chunk, device)
            optimizer.zero_grad()
            logits = model(big, owner, lengths, device)
            loss = F.cross_entropy(logits, labels, weight=class_weight)
            loss.backward()
            optimizer.step()

        val_f1 = evaluate(model, val_g, 256, device)["f1_macro"]
        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1, best_state, no_improve = val_f1, deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_g, 256, device)
    return {"config": config, "best_val_f1_macro": best_val_f1, "test_metrics": test_metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=6)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    rng = random.Random(RANDOM_STATE)

    print("Cargando dataset GNN...")
    dataset = torch.load(DATASET_PATH, weights_only=False)
    input_dim = int(dataset[0].x.shape[1])
    games = build_games(dataset)
    num_classes = len({int(s[0].y.item()) for s in games.values()})
    print(f"  partidas={len(games)}  input_dim={input_dim}  classes={num_classes}")

    train_g, val_g, test_g = stratified_game_split(games)
    print(f"  partidas: train={len(train_g)} val={len(val_g)} test={len(test_g)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}")

    train_labels = np.array([int(s[0].y.item()) for s in train_g])
    counts = np.bincount(train_labels, minlength=num_classes).astype(float)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    class_weight = torch.tensor(weights, dtype=torch.float, device=device)
    print(f"  class_weight={weights.round(3).tolist()}")

    sampled, seen = [], set()
    while len(sampled) < args.n_trials:
        c = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
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
                config, train_g, val_g, test_g, input_dim, num_classes,
                class_weight, args.max_epochs, args.patience, device, rng,
            )
        res["elapsed_seconds"] = round(t.elapsed, 2)
        print(
            f"  val game f1={res['best_val_f1_macro']:.4f} "
            f"TEST game f1={res['test_metrics']['f1_macro']:.4f} "
            f"acc={res['test_metrics']['accuracy']:.4f} t={res['elapsed_seconds']}s"
        )
        trials.append(res)

    best = max(trials, key=lambda r: r["best_val_f1_macro"])
    payload = {
        "model": "gnn_temporal",
        "features": "graphs",
        "search_kind": "random",
        "evaluation_unit": "game (GRU over snapshot sequence)",
        "n_train": len(train_g),
        "n_test": len(test_g),
        "search_space": SEARCH_SPACE,
        "n_trials": args.n_trials,
        "all_trials": trials,
        "best_run": {
            "model": "gnn_temporal",
            "features": "graphs",
            "search_kind": "random",
            "n_train": len(train_g),
            "n_test": len(test_g),
            "elapsed_seconds": best["elapsed_seconds"],
            "best_params": best["config"],
            "best_cv_f1_macro": best["best_val_f1_macro"],
            "test_metrics": best["test_metrics"],
        },
    }
    path = save_result("gnn_temporal_graphs", payload)
    print(f"\n=> guardado en {path}")
    print(
        f"BEST: val game f1={best['best_val_f1_macro']:.4f}  "
        f"test game f1={best['test_metrics']['f1_macro']:.4f}"
    )


if __name__ == "__main__":
    main()

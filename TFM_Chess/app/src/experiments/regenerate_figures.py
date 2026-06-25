"""
Genera las figuras de §4 a partir de los JSON guardados en app/results/hp_search/.

Produce:
- una matriz de confusión por modelo  -> app/results/figures/cm_{model}_{features}.png
- gráfico de barras comparando F1-macro por modelo -> app/results/figures/comparison_f1_macro.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HP_DIR = Path("app/results/hp_search")
FIG_DIR = Path("app/results/figures")


def plot_confusion(cm: list[list[int]], labels: list[str], title: str, out: Path):
    arr = np.array(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(title)
    threshold = arr.max() / 2 if arr.max() else 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(
                j, i, arr[i, j],
                ha="center", va="center",
                color="white" if arr[i, j] > threshold else "black",
                fontsize=10,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_comparison_bars(rows: list[dict], out: Path):
    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(df))
    ax.barh(y, df["f1_macro"], color="steelblue")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.modelo} / {r.features}" for r in df.itertuples()])
    ax.set_xlabel("F1-macro (test)")
    ax.set_title("Comparativa de modelos por F1-macro")
    for i, v in enumerate(df["f1_macro"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(HP_DIR.glob("*.json")):
        payload = json.load(open(path, encoding="utf-8"))
        best = payload["best_run"]
        cm = best["test_metrics"]["confusion_matrix"]
        title = (
            f"{payload['model']} · {payload['features']}\n"
            f"F1-macro={best['test_metrics']['f1_macro']:.3f} · "
            f"acc={best['test_metrics']['accuracy']:.3f}"
        )
        out = FIG_DIR / f"cm_{payload['model']}_{payload['features']}.png"
        plot_confusion(cm["matrix"], cm["labels"], title, out)
        rows.append(
            {
                "modelo": payload["model"],
                "features": payload["features"],
                "f1_macro": best["test_metrics"]["f1_macro"],
                "accuracy": best["test_metrics"]["accuracy"],
            }
        )
        print(f"-> {out}")

    if rows:
        bars = FIG_DIR / "comparison_f1_macro.png"
        plot_comparison_bars(rows, bars)
        print(f"-> {bars}")


if __name__ == "__main__":
    main()

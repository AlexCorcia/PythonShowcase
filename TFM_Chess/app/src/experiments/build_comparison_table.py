"""
Construye la tabla consolidada de comparación entre modelos,
ordenada por F1-macro (criterio principal según las indicaciones del director).

Lee app/results/hp_search/*.json y produce:
- app/results/tables/final_comparison.csv  (datos)
- app/results/tables/final_comparison.md   (mismo contenido en markdown)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HP_DIR = Path("app/results/hp_search")
OUT_DIR = Path("app/results/tables")


def _format_params(params: dict | None) -> str:
    if not params:
        return ""
    parts = []
    for k, v in params.items():
        clean_key = k.split("__", 1)[1] if "__" in k else k
        parts.append(f"{clean_key}={v}")
    return ", ".join(parts)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(HP_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        best = payload.get("best_run", {})
        test_metrics = best.get("test_metrics", {})

        rows.append(
            {
                "modelo": payload.get("model"),
                "features": payload.get("features"),
                "search_kind": best.get("search_kind"),
                "best_params": _format_params(best.get("best_params")),
                "cv_f1_macro": round(best.get("best_cv_f1_macro", float("nan")), 4),
                "test_accuracy": round(test_metrics.get("accuracy", float("nan")), 4),
                "test_f1_macro": round(test_metrics.get("f1_macro", float("nan")), 4),
                "test_f1_weighted": round(
                    test_metrics.get("f1_weighted", float("nan")), 4
                ),
                "n_train": best.get("n_train"),
                "n_test": best.get("n_test"),
                "elapsed_s": best.get("elapsed_seconds"),
                "source_file": path.name,
            }
        )

    if not rows:
        print("No se encontraron archivos en", HP_DIR)
        return

    df = pd.DataFrame(rows).sort_values(
        by="test_f1_macro", ascending=False
    ).reset_index(drop=True)

    csv_path = OUT_DIR / "final_comparison.csv"
    df.to_csv(csv_path, index=False)

    md_path = OUT_DIR / "final_comparison.md"
    cols_for_md = [
        "modelo",
        "features",
        "search_kind",
        "cv_f1_macro",
        "test_accuracy",
        "test_f1_macro",
        "test_f1_weighted",
        "elapsed_s",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Comparativa global de modelos (ordenada por F1-macro de test)\n\n")
        f.write(df[cols_for_md].to_markdown(index=False))
        f.write("\n\n## Mejores hiperparámetros por modelo\n\n")
        for _, r in df.iterrows():
            f.write(
                f"- **{r['modelo']} / {r['features']}** "
                f"({r['search_kind']}, F1-macro test={r['test_f1_macro']:.4f}): "
                f"`{r['best_params']}`\n"
            )

    print(f"-> {csv_path}")
    print(f"-> {md_path}")
    print("\nResumen:")
    print(df[cols_for_md].to_string(index=False))


if __name__ == "__main__":
    main()

import json
from pathlib import Path
import pandas as pd


input_path = Path("app/data/graphs/all_players_graphs.json")
output_dir = Path("app/results/tables")
output_dir.mkdir(parents=True, exist_ok=True)

with open(input_path, "r", encoding="utf-8") as f:
    graphs = json.load(f)

rows = []

for game in graphs:
    metadata = game["metadata"]
    snapshots = game["snapshots"]

    num_snapshots = len(snapshots)
    avg_nodes = sum(len(s["nodes"]) for s in snapshots) / num_snapshots
    avg_edges = sum(len(s["edges"]) for s in snapshots) / num_snapshots

    rows.append({
        "main_player": metadata["main_player"],
        "style": metadata["style"],
        "eco": metadata["eco"],
        "num_snapshots": num_snapshots,
        "avg_nodes": avg_nodes,
        "avg_edges": avg_edges,
    })

df = pd.DataFrame(rows)

summary = (
    df.groupby(["main_player", "style"])
    [["num_snapshots", "avg_nodes", "avg_edges"]]
    .mean()
    .round(2)
)

print(summary)

summary.to_csv(output_dir / "graph_dataset_summary.csv")

print("\nResumen guardado en:")
print(output_dir / "graph_dataset_summary.csv")
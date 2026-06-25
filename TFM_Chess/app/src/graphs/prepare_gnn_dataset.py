from pathlib import Path
import json

import torch
from torch_geometric.data import Data


STYLE_TO_LABEL = {
    "defensive": 0,
    "dynamic": 1,
    "positional": 2,
    "tactical": 3,
}

PIECE_TO_ID = {
    "pawn": 1,
    "knight": 2,
    "bishop": 3,
    "rook": 4,
    "queen": 5,
    "king": 6,
}

COLOR_TO_ID = {
    "white": 0,
    "black": 1,
}


def node_to_features(node: dict) -> list[float]:
    """
    Convierte un nodo pieza en vector numérico.
    """
    piece_id = PIECE_TO_ID.get(node["piece_name"], 0)
    color_id = COLOR_TO_ID.get(node["color"], 0)

    return [
        piece_id,
        color_id,
        node["x"] / 7,
        node["y"] / 7,
        node["value"] / 9,
    ]


def snapshot_to_pyg_data(snapshot: dict, style: str) -> Data:
    """
    Convierte un snapshot de grafo a objeto PyTorch Geometric.
    """
    nodes = snapshot["nodes"]
    edges = snapshot["edges"]

    node_id_to_index = {
        node["id"]: idx
        for idx, node in enumerate(nodes)
    }

    x = torch.tensor(
        [node_to_features(node) for node in nodes],
        dtype=torch.float,
    )

    edge_index_list = []

    for edge in edges:
        source_id = edge["source"]
        target_id = edge["target"]

        if source_id not in node_id_to_index:
            continue

        if target_id not in node_id_to_index:
            continue

        source_idx = node_id_to_index[source_id]
        target_idx = node_id_to_index[target_id]

        edge_index_list.append([source_idx, target_idx])

    if len(edge_index_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(
            edge_index_list,
            dtype=torch.long,
        ).t().contiguous()

    y = torch.tensor(
        [STYLE_TO_LABEL[style]],
        dtype=torch.long,
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
    )

    return data


def build_gnn_dataset():
    input_path = Path("app/data/graphs/all_players_graphs.json")
    output_path = Path("app/data/graphs/graph_dataset.pt")

    if not input_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {input_path}"
        )

    with open(input_path, "r", encoding="utf-8") as f:
        graph_games = json.load(f)

    dataset = []

    for game in graph_games:
        metadata = game["metadata"]
        style = metadata["style"]

        for snapshot in game["snapshots"]:
            data = snapshot_to_pyg_data(
                snapshot=snapshot,
                style=style,
            )

            data.main_player = metadata["main_player"]
            data.style = style
            data.ply_number = snapshot["ply_number"]
            data.eco = metadata.get("eco")

            dataset.append(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(dataset, output_path)

    print("\n======================================")
    print("Dataset GNN generado correctamente")
    print("======================================")
    print(f"Número de snapshots: {len(dataset)}")
    print(f"Archivo guardado en: {output_path}")

    print("\nEjemplo primer grafo:")
    print(dataset[0])
    print("Node feature shape:", dataset[0].x.shape)
    print("Edge index shape:", dataset[0].edge_index.shape)
    print("Label:", dataset[0].y.item())


if __name__ == "__main__":
    build_gnn_dataset()
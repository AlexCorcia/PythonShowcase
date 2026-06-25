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

# Tipos de pieza para codificación one-hot.
PIECE_ORDER = ["pawn", "knight", "bishop", "rook", "queen", "king"]
PIECE_TO_INDEX = {name: i for i, name in enumerate(PIECE_ORDER)}

# Dimensión del vector de características de nodo:
#   6 (one-hot pieza) + 2 (one-hot color) + 2 (x, y) + 1 (valor)
#   + 2 (grado de ataque y de defensa normalizados) = 13
NODE_FEATURE_DIM = len(PIECE_ORDER) + 2 + 2 + 1 + 2


def compute_node_degrees(nodes: list[dict], edges: list[dict]) -> dict[int, tuple[int, int]]:
    """
    Para cada nodo, cuenta cuántas aristas de ataque y de defensa salen de él.
    Capta la actividad/conectividad de cada pieza en la posición.
    """
    degrees = {node["id"]: [0, 0] for node in nodes}
    for edge in edges:
        src = edge["source"]
        if src not in degrees:
            continue
        if edge["relation"] == "attack":
            degrees[src][0] += 1
        else:
            degrees[src][1] += 1
    return {k: (v[0], v[1]) for k, v in degrees.items()}


def node_to_features(node: dict, degrees: tuple[int, int]) -> list[float]:
    """
    Vector numérico de un nodo pieza, con codificación one-hot del tipo de pieza
    y del color (evita que la GCN interprete el tipo de pieza como una escala
    ordinal), más posición, valor y grados de ataque/defensa.
    """
    piece_onehot = [0.0] * len(PIECE_ORDER)
    idx = PIECE_TO_INDEX.get(node["piece_name"])
    if idx is not None:
        piece_onehot[idx] = 1.0

    color_onehot = [1.0, 0.0] if node["color"] == "white" else [0.0, 1.0]

    attack_deg, defense_deg = degrees
    # Normalización suave (el grado máximo realista de una pieza es ~13-27).
    attack_norm = attack_deg / 8.0
    defense_norm = defense_deg / 8.0

    return (
        piece_onehot
        + color_onehot
        + [node["x"] / 7.0, node["y"] / 7.0]
        + [node["value"] / 9.0]
        + [attack_norm, defense_norm]
    )


def snapshot_to_pyg_data(snapshot: dict, style: str) -> Data:
    nodes = snapshot["nodes"]
    edges = snapshot["edges"]

    node_id_to_index = {node["id"]: idx for idx, node in enumerate(nodes)}
    degrees = compute_node_degrees(nodes, edges)

    x = torch.tensor(
        [node_to_features(node, degrees[node["id"]]) for node in nodes],
        dtype=torch.float,
    )

    edge_index_list = []
    for edge in edges:
        source_id = edge["source"]
        target_id = edge["target"]
        if source_id not in node_id_to_index or target_id not in node_id_to_index:
            continue
        edge_index_list.append(
            [node_id_to_index[source_id], node_id_to_index[target_id]]
        )

    if len(edge_index_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    y = torch.tensor([STYLE_TO_LABEL[style]], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


def build_gnn_dataset():
    input_path = Path("app/data/graphs/all_players_graphs.json")
    output_path = Path("app/data/graphs/graph_dataset.pt")

    if not input_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        graph_games = json.load(f)

    dataset = []

    for game in graph_games:
        metadata = game["metadata"]
        style = metadata["style"]
        game_index = metadata.get("game_index", -1)

        for snapshot in game["snapshots"]:
            data = snapshot_to_pyg_data(snapshot=snapshot, style=style)

            # Metadatos para partición por partida y agregación a nivel de partida.
            data.game_index = int(game_index)
            data.main_player = metadata["main_player"]
            data.style = style
            data.ply_number = int(snapshot["ply_number"])
            data.phase = snapshot.get("phase", "")
            data.eco = metadata.get("eco")

            dataset.append(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)

    n_games = len({d.game_index for d in dataset})
    print("\n======================================")
    print("Dataset GNN generado correctamente")
    print("======================================")
    print(f"Número de snapshots: {len(dataset)}")
    print(f"Número de partidas: {n_games}")
    print(f"Dimensión de características de nodo: {dataset[0].x.shape[1]}")
    print("\nEjemplo primer grafo:")
    print(dataset[0])
    print("Label:", dataset[0].y.item())


if __name__ == "__main__":
    build_gnn_dataset()

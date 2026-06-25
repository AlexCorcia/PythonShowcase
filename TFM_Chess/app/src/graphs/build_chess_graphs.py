from pathlib import Path
import json
import sys

import chess
import chess.pgn

sys.path.append("app/src/preprocessing")

from parse_pgn import detect_player_perspective


PLAYERS = {
    "Karpov": {"file": "Karpov.pgn", "style": "positional"},
    "Tal": {"file": "Tal.pgn", "style": "tactical"},
    "Kasparov": {"file": "Kasparov.pgn", "style": "dynamic"},
    "Petrosian": {"file": "Petrosian.pgn", "style": "defensive"},
}


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def square_to_coordinates(square: int) -> tuple[int, int]:
    return chess.square_file(square), chess.square_rank(square)


def build_graph_snapshot(board: chess.Board, ply_number: int) -> dict:
    nodes = []
    square_to_node_id = {}

    for node_id, square in enumerate(chess.SQUARES):
        piece = board.piece_at(square)

        if piece is None:
            continue

        x, y = square_to_coordinates(square)

        node = {
            "id": node_id,
            "square": chess.square_name(square),
            "x": x,
            "y": y,
            "piece_type": piece.symbol().lower(),
            "piece_name": chess.piece_name(piece.piece_type),
            "color": "white" if piece.color == chess.WHITE else "black",
            "value": PIECE_VALUES[piece.piece_type],
        }

        square_to_node_id[square] = node_id
        nodes.append(node)

    edges = []

    for source_square, source_node_id in square_to_node_id.items():
        source_piece = board.piece_at(source_square)

        if source_piece is None:
            continue

        attacked_squares = board.attacks(source_square)

        for target_square in attacked_squares:
            if target_square not in square_to_node_id:
                continue

            target_piece = board.piece_at(target_square)

            if target_piece is None:
                continue

            relation = (
                "attack"
                if source_piece.color != target_piece.color
                else "defense"
            )

            edges.append(
                {
                    "source": source_node_id,
                    "target": square_to_node_id[target_square],
                    "relation": relation,
                    "source_square": chess.square_name(source_square),
                    "target_square": chess.square_name(target_square),
                }
            )

    return {
        "ply_number": ply_number,
        "turn": "white" if board.turn == chess.WHITE else "black",
        "nodes": nodes,
        "edges": edges,
    }


def process_game(game: chess.pgn.Game, main_player: str, style: str, max_snapshots: int = 20) -> dict | None:
    headers = game.headers

    white = headers.get("White", "").strip()
    black = headers.get("Black", "").strip()

    player_color, opponent = detect_player_perspective(
        white=white,
        black=black,
        main_player=main_player,
    )

    if player_color is None:
        return None

    board = game.board()

    snapshots = []

    snapshots.append(
        build_graph_snapshot(
            board=board,
            ply_number=0,
        )
    )

    for ply_number, move in enumerate(game.mainline_moves(), start=1):
        board.push(move)

        if ply_number <= max_snapshots:
            snapshots.append(
                build_graph_snapshot(
                    board=board,
                    ply_number=ply_number,
                )
            )

    graph_game = {
        "metadata": {
            "main_player": main_player,
            "style": style,
            "player_color": player_color,
            "opponent": opponent,
            "white": white,
            "black": black,
            "event": headers.get("Event", ""),
            "site": headers.get("Site", ""),
            "date": headers.get("Date", ""),
            "result": headers.get("Result", ""),
            "eco": headers.get("ECO", None),
        },
        "snapshots": snapshots,
    }

    return graph_game


def build_graph_dataset(max_games_per_player: int = 50, max_snapshots: int = 20):
    raw_dir = Path("app/data/raw")
    output_dir = Path("app/data/graphs")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_graphs = []

    for player_name, config in PLAYERS.items():
        pgn_path = raw_dir / config["file"]

        if not pgn_path.exists():
            print(f"Archivo no encontrado: {pgn_path}")
            continue

        print(f"\nProcesando grafos de: {player_name}")

        player_graphs = []
        total_games = 0
        valid_graphs = 0

        with open(pgn_path, "r", encoding="utf-8", errors="ignore") as pgn:
            while valid_graphs < max_games_per_player:
                game = chess.pgn.read_game(pgn)

                if game is None:
                    break

                total_games += 1

                graph_game = process_game(
                    game=game,
                    main_player=player_name,
                    style=config["style"],
                    max_snapshots=max_snapshots,
                )

                if graph_game is None:
                    continue

                player_graphs.append(graph_game)
                all_graphs.append(graph_game)

                valid_graphs += 1

        player_output = output_dir / f"{player_name.lower()}_graphs.json"

        with open(player_output, "w", encoding="utf-8") as f:
            json.dump(player_graphs, f, indent=2, ensure_ascii=False)

        print(f"Partidas leídas: {total_games}")
        print(f"Grafos generados: {valid_graphs}")
        print(f"Archivo guardado: {player_output}")

    full_output = output_dir / "all_players_graphs.json"

    with open(full_output, "w", encoding="utf-8") as f:
        json.dump(all_graphs, f, indent=2, ensure_ascii=False)

    print("\n======================================")
    print("Dataset de grafos generado")
    print(f"Total partidas grafo: {len(all_graphs)}")
    print(f"Archivo: {full_output}")
    print("======================================")


if __name__ == "__main__":
    build_graph_dataset(
        max_games_per_player=50,
        max_snapshots=20,
    )
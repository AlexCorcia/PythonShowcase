from pathlib import Path
import argparse
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


def build_graph_snapshot(board: chess.Board, ply_number: int, total_plies: int) -> dict:
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

    # Fase de la partida en la que se toma el snapshot (0=apertura, 1=medio, 2=final).
    # Útil como rasgo de grafo y para verificar la cobertura de fases.
    frac = ply_number / total_plies if total_plies else 0.0
    if frac < 1 / 3:
        phase = "opening"
    elif frac < 2 / 3:
        phase = "middlegame"
    else:
        phase = "endgame"

    return {
        "ply_number": ply_number,
        "phase": phase,
        "turn": "white" if board.turn == chess.WHITE else "black",
        "nodes": nodes,
        "edges": edges,
    }


def select_snapshot_plies(total_plies: int, snapshots_per_game: int) -> list[int]:
    """
    Selecciona índices de ply repartidos uniformemente a lo largo de TODA la
    partida (apertura, medio juego y final), evitando la posición inicial
    (ply 0), que es idéntica en todas las partidas.

    Para partidas más cortas que `snapshots_per_game`, devuelve todos los plies.
    """
    if total_plies <= 0:
        return []
    if total_plies <= snapshots_per_game:
        return list(range(1, total_plies + 1))

    # Posiciones equiespaciadas en (0, total_plies], excluyendo el ply 0.
    plies = []
    for k in range(1, snapshots_per_game + 1):
        ply = round(k * total_plies / snapshots_per_game)
        ply = max(1, min(total_plies, ply))
        if ply not in plies:
            plies.append(ply)
    return plies


def process_game(
    game: chess.pgn.Game,
    main_player: str,
    style: str,
    game_index: int,
    snapshots_per_game: int = 12,
) -> dict | None:
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

    moves = list(game.mainline_moves())
    total_plies = len(moves)

    if total_plies < 6:
        # Partidas demasiado cortas (abandonos, datos incompletos) no aportan.
        return None

    target_plies = set(select_snapshot_plies(total_plies, snapshots_per_game))

    board = game.board()
    snapshots = []

    for ply_number, move in enumerate(moves, start=1):
        board.push(move)
        if ply_number in target_plies:
            snapshots.append(
                build_graph_snapshot(
                    board=board,
                    ply_number=ply_number,
                    total_plies=total_plies,
                )
            )

    if not snapshots:
        return None

    graph_game = {
        "metadata": {
            "game_index": game_index,
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
            "total_plies": total_plies,
        },
        "snapshots": snapshots,
    }

    return graph_game


def build_graph_dataset(max_games_per_player: int | None = None, snapshots_per_game: int = 12):
    raw_dir = Path("app/data/raw")
    output_dir = Path("app/data/graphs")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_graphs = []
    game_index = 0

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
            while max_games_per_player is None or valid_graphs < max_games_per_player:
                game = chess.pgn.read_game(pgn)

                if game is None:
                    break

                total_games += 1

                graph_game = process_game(
                    game=game,
                    main_player=player_name,
                    style=config["style"],
                    game_index=game_index,
                    snapshots_per_game=snapshots_per_game,
                )

                if graph_game is None:
                    continue

                player_graphs.append(graph_game)
                all_graphs.append(graph_game)

                valid_graphs += 1
                game_index += 1

        player_output = output_dir / f"{player_name.lower()}_graphs.json"

        with open(player_output, "w", encoding="utf-8") as f:
            json.dump(player_graphs, f, ensure_ascii=False)

        print(f"Partidas leídas: {total_games}")
        print(f"Grafos generados: {valid_graphs}")
        print(f"Archivo guardado: {player_output}")

    full_output = output_dir / "all_players_graphs.json"

    with open(full_output, "w", encoding="utf-8") as f:
        json.dump(all_graphs, f, ensure_ascii=False)

    total_snapshots = sum(len(g["snapshots"]) for g in all_graphs)

    print("\n======================================")
    print("Dataset de grafos generado")
    print(f"Total partidas grafo: {len(all_graphs)}")
    print(f"Total snapshots: {total_snapshots}")
    print(f"Archivo: {full_output}")
    print("======================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-games-per-player",
        type=int,
        default=0,
        help="0 = usar todas las partidas disponibles",
    )
    parser.add_argument("--snapshots-per-game", type=int, default=12)
    args = parser.parse_args()

    build_graph_dataset(
        max_games_per_player=None if args.max_games_per_player == 0 else args.max_games_per_player,
        snapshots_per_game=args.snapshots_per_game,
    )

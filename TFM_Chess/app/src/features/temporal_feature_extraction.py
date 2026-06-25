from pathlib import Path
import sys

import chess
import chess.pgn
import pandas as pd

sys.path.append("app/src/preprocessing")

from parse_pgn import (
    detect_player_perspective,
    parse_player_result,
    parse_result,
    extract_year,
    clean_elo,
)


PLAYERS = {
    "Karpov": {"file": "Karpov.pgn", "style": "positional"},
    "Tal": {"file": "Tal.pgn", "style": "tactical"},
    "Kasparov": {"file": "Kasparov.pgn", "style": "dynamic"},
    "Petrosian": {"file": "Petrosian.pgn", "style": "defensive"},
}


def get_phase(ply_number: int) -> str:
    if ply_number <= 20:
        return "opening"
    if ply_number <= 60:
        return "middlegame"
    return "endgame"


def empty_phase_features() -> dict:
    phases = ["opening", "middlegame", "endgame"]
    features = {}

    for phase in phases:
        features[f"{phase}_plies"] = 0
        features[f"{phase}_player_captures"] = 0
        features[f"{phase}_player_checks"] = 0
        features[f"{phase}_player_castles"] = 0
        features[f"{phase}_player_promotions"] = 0
        features[f"{phase}_opponent_captures"] = 0
        features[f"{phase}_opponent_checks"] = 0

    return features


def count_temporal_features(game, player_color: str) -> dict:
    features = empty_phase_features()

    board = game.board()
    player_is_white = player_color == "white"

    ply_number = 0

    for move in game.mainline_moves():
        ply_number += 1
        phase = get_phase(ply_number)

        moving_color = board.turn

        is_player_move = (
            moving_color == chess.WHITE and player_is_white
        ) or (
            moving_color == chess.BLACK and not player_is_white
        )

        features[f"{phase}_plies"] += 1

        if board.is_capture(move):
            if is_player_move:
                features[f"{phase}_player_captures"] += 1
            else:
                features[f"{phase}_opponent_captures"] += 1

        if board.is_castling(move) and is_player_move:
            features[f"{phase}_player_castles"] += 1

        if move.promotion is not None and is_player_move:
            features[f"{phase}_player_promotions"] += 1

        board.push(move)

        if board.is_check():
            if is_player_move:
                features[f"{phase}_player_checks"] += 1
            else:
                features[f"{phase}_opponent_checks"] += 1

    return features


def parse_temporal_player_file(
    pgn_path: Path,
    main_player: str,
    style: str,
    min_plies: int = 20,
) -> pd.DataFrame:

    rows = []

    total_games = 0
    valid_games = 0
    skipped_games = 0

    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as pgn:

        while True:
            game = chess.pgn.read_game(pgn)

            if game is None:
                break

            total_games += 1

            headers = game.headers

            white = headers.get("White", "").strip()
            black = headers.get("Black", "").strip()
            result = headers.get("Result", "").strip()

            player_color, opponent = detect_player_perspective(
                white=white,
                black=black,
                main_player=main_player,
            )

            if player_color is None:
                skipped_games += 1
                continue

            moves = list(game.mainline_moves())
            num_plies = len(moves)

            if num_plies < min_plies:
                skipped_games += 1
                continue

            try:
                temporal_features = count_temporal_features(
                    game=game,
                    player_color=player_color,
                )
            except Exception:
                skipped_games += 1
                continue

            eco = headers.get("ECO", None)

            row = {
                "source_file": pgn_path.name,
                "main_player": main_player,
                "style": style,
                "player_color": player_color,
                "opponent": opponent,

                "event": headers.get("Event", ""),
                "site": headers.get("Site", ""),
                "date": headers.get("Date", ""),
                "year": extract_year(headers.get("Date", "")),
                "round": headers.get("Round", ""),

                "white": white,
                "black": black,
                "white_elo": clean_elo(headers.get("WhiteElo")),
                "black_elo": clean_elo(headers.get("BlackElo")),

                "result": result,
                "result_label": parse_result(result),
                "player_result": parse_player_result(result, player_color),

                "eco": eco,
                "eco_family": str(eco)[0] if eco else None,

                "num_plies": num_plies,
                "num_moves": num_plies / 2,
            }

            row.update(temporal_features)

            rows.append(row)
            valid_games += 1

    df = pd.DataFrame(rows)

    print("=" * 60)
    print(f"Archivo: {pgn_path}")
    print(f"Jugador: {main_player}")
    print(f"Partidas leídas: {total_games}")
    print(f"Partidas válidas: {valid_games}")
    print(f"Partidas omitidas: {skipped_games}")
    print("=" * 60)

    return df


def build_temporal_dataset():
    raw_dir = Path("app/data/raw")
    output_dir = Path("app/data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []

    for player_name, config in PLAYERS.items():
        pgn_path = raw_dir / config["file"]

        if not pgn_path.exists():
            print(f"Archivo no encontrado: {pgn_path}")
            continue

        df_player = parse_temporal_player_file(
            pgn_path=pgn_path,
            main_player=player_name,
            style=config["style"],
            min_plies=20,
        )

        all_dfs.append(df_player)

    temporal_df = pd.concat(all_dfs, ignore_index=True)

    valid_eco = ["A", "B", "C", "D", "E"]
    temporal_df = temporal_df[
        temporal_df["eco_family"].isin(valid_eco)
    ]

    output_path = output_dir / "master_games_temporal.csv"

    temporal_df.to_csv(output_path, index=False)

    print("\n====================================")
    print("Dataset temporal generado")
    print(f"Shape: {temporal_df.shape}")
    print(f"Archivo: {output_path}")
    print("====================================")

    print("\nPartidas por estilo:")
    print(temporal_df["style"].value_counts())


if __name__ == "__main__":
    build_temporal_dataset()
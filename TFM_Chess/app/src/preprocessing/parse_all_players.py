from pathlib import Path
import pandas as pd

from parse_pgn import parse_pgn_file


PLAYERS = {
    "Karpov": {
        "file": "Karpov.pgn",
        "style": "positional",
    },
    "Tal": {
        "file": "Tal.pgn",
        "style": "tactical",
    },
    "Kasparov": {
        "file": "Kasparov.pgn",
        "style": "dynamic",
    },
    "Petrosian": {
        "file": "Petrosian.pgn",
        "style": "defensive",
    },
}


def parse_all_players():
    raw_dir = Path("app/data/raw")
    output_dir = Path("app/data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []

    for player_name, config in PLAYERS.items():
        pgn_path = raw_dir / config["file"]

        if not pgn_path.exists():
            print(f"Archivo no encontrado: {pgn_path}")
            continue

        print(f"\nProcesando jugador: {player_name}")

        df_player = parse_pgn_file(
            pgn_path=pgn_path,
            main_player=player_name,
            min_plies=20,
        )

        df_player["style"] = config["style"]

        individual_output = output_dir / f"{player_name.lower()}_parsed.csv"
        df_player.to_csv(individual_output, index=False)

        print(f"CSV individual guardado en: {individual_output}")

        all_dfs.append(df_player)

    if not all_dfs:
        print("No se procesó ningún jugador.")
        return

    master_df = pd.concat(all_dfs, ignore_index=True)

    master_output = output_dir / "master_games.csv"
    master_df.to_csv(master_output, index=False)

    print("\n========================================")
    print("Dataset global generado correctamente")
    print(f"Total partidas: {len(master_df)}")
    print(f"Archivo: {master_output}")
    print("========================================")

    print("\nPartidas por jugador:")
    print(master_df["main_player"].value_counts())

    print("\nPartidas por estilo:")
    print(master_df["style"].value_counts())


if __name__ == "__main__":
    parse_all_players()
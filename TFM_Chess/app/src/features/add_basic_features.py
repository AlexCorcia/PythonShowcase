from pathlib import Path

import pandas as pd


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["capture_rate"] = df["player_captures"] / df["num_moves"]
    df["check_rate"] = df["player_checks"] / df["num_moves"]
    df["aggression_score"] = df["capture_rate"] + df["check_rate"]

    return df


def main():
    input_path = Path("app/data/processed/master_games.csv")
    output_path = Path("app/data/processed/master_games_enriched.csv")

    df = pd.read_csv(input_path)
    print(f"Cargado: {df.shape}")

    df = add_basic_features(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Guardado: {output_path} ({df.shape})")
    print("\nResumen de las nuevas variables:")
    print(df[["capture_rate", "check_rate", "aggression_score"]].describe().round(4))


if __name__ == "__main__":
    main()

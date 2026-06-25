from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("app/data/processed/master_games_enriched.csv")

figures_dir = Path("app/results/figures")
figures_dir.mkdir(parents=True, exist_ok=True)

# ==============================
# 1. Partidas por jugador
# ==============================

plt.figure(figsize=(8, 5))
df["main_player"].value_counts().plot(kind="bar")
plt.title("Número de partidas por jugador")
plt.xlabel("Jugador")
plt.ylabel("Número de partidas")
plt.tight_layout()
plt.savefig(figures_dir / "games_by_player.png")
plt.close()


# ==============================
# 2. Partidas por estilo
# ==============================

plt.figure(figsize=(8, 5))
df["style"].value_counts().plot(kind="bar")
plt.title("Número de partidas por estilo")
plt.xlabel("Estilo")
plt.ylabel("Número de partidas")
plt.tight_layout()
plt.savefig(figures_dir / "games_by_style.png")
plt.close()


# ==============================
# 3. Resultados por jugador
# ==============================

results = pd.crosstab(
    df["main_player"],
    df["player_result"],
    normalize="index"
) * 100

results.plot(kind="bar", figsize=(9, 5))
plt.title("Distribución de resultados por jugador")
plt.xlabel("Jugador")
plt.ylabel("Porcentaje")
plt.legend(title="Resultado")
plt.tight_layout()
plt.savefig(figures_dir / "results_by_player.png")
plt.close()


# ==============================
# 4. Capturas medias por jugador
# ==============================

captures = df.groupby("main_player")["player_captures"].mean().sort_values()

plt.figure(figsize=(8, 5))
captures.plot(kind="bar")
plt.title("Capturas medias por jugador")
plt.xlabel("Jugador")
plt.ylabel("Capturas medias")
plt.tight_layout()
plt.savefig(figures_dir / "avg_captures_by_player.png")
plt.close()


# ==============================
# 5. Jaques medios por jugador
# ==============================

checks = df.groupby("main_player")["player_checks"].mean().sort_values()

plt.figure(figsize=(8, 5))
checks.plot(kind="bar")
plt.title("Jaques medios por jugador")
plt.xlabel("Jugador")
plt.ylabel("Jaques medios")
plt.tight_layout()
plt.savefig(figures_dir / "avg_checks_by_player.png")
plt.close()


# ==============================
# 6. Agresividad media por jugador
# ==============================

aggression = df.groupby("main_player")["aggression_score"].mean().sort_values()

plt.figure(figsize=(8, 5))
aggression.plot(kind="bar")
plt.title("Agresividad media por jugador")
plt.xlabel("Jugador")
plt.ylabel("Aggression score")
plt.tight_layout()
plt.savefig(figures_dir / "aggression_by_player.png")
plt.close()


# ==============================
# 7. Duración media por jugador
# ==============================

duration = df.groupby("main_player")["num_moves"].mean().sort_values()

plt.figure(figsize=(8, 5))
duration.plot(kind="bar")
plt.title("Duración media de las partidas por jugador")
plt.xlabel("Jugador")
plt.ylabel("Número medio de movimientos")
plt.tight_layout()
plt.savefig(figures_dir / "avg_game_length_by_player.png")
plt.close()


print("Gráficas guardadas en:", figures_dir)
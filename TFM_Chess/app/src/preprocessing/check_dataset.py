import pandas as pd


# ============================================
# Cargar dataset
# ============================================

df = pd.read_csv("app/data/processed/karpov_parsed.csv")


# ============================================
# Información general
# ============================================

print("\n================ DATASET INFO ================\n")

print("Shape del dataset:")
print(df.shape)

print("\nColumnas:")
print(df.columns.tolist())


# ============================================
# Valores nulos
# ============================================

print("\n================ NULL VALUES ================\n")

print(df.isna().sum().sort_values(ascending=False))


# ============================================
# Distribución colores
# ============================================

print("\n================ PLAYER COLORS ================\n")

print(df["player_color"].value_counts())


# ============================================
# Distribución resultados
# ============================================

print("\n================ PLAYER RESULTS ================\n")

print(df["player_result"].value_counts())


# ============================================
# Estadísticas movimientos
# ============================================

print("\n================ NUM MOVES STATS ================\n")

print(df["num_moves"].describe())


# ============================================
# Estadísticas capturas
# ============================================

print("\n================ CAPTURES STATS ================\n")

print(df["player_captures"].describe())


# ============================================
# Años disponibles
# ============================================

print("\n================ YEAR STATS ================\n")

print(df["year"].describe())


# ============================================
# Aperturas más frecuentes
# ============================================

print("\n================ TOP OPENINGS ================\n")

print(df["opening"].value_counts().head(20))


# ============================================
# ECO más frecuentes
# ============================================

print("\n================ TOP ECO CODES ================\n")

print(df["eco"].value_counts().head(20))


# ============================================
# Comprobación visual
# ============================================

print("\n================ SAMPLE ROWS ================\n")

columns_to_show = [
    "main_player",
    "player_color",
    "opponent",
    "result",
    "player_result",
    "eco",
    "opening",
    "num_moves",
    "player_captures",
    "player_checks",
]

print(df[columns_to_show].head(20))
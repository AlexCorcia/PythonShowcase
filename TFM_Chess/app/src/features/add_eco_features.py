from pathlib import Path

import pandas as pd


# ============================================
# Cargar dataset enriquecido
# ============================================

input_path = Path(
    "app/data/processed/master_games_enriched.csv"
)

df = pd.read_csv(input_path)

print("\nDataset cargado:")
print(df.shape)


# ============================================
# Crear eco_family
# ============================================

df["eco_family"] = df["eco"].astype(str).str[0]

print("\nDistribución ECO families:\n")

print(
    df["eco_family"]
    .value_counts()
)


# ============================================
# Validar valores
# ============================================

valid_eco = ["A", "B", "C", "D", "E"]

df = df[
    df["eco_family"].isin(valid_eco)
]

print("\nDataset tras validar ECO:")
print(df.shape)


# ============================================
# Guardar dataset actualizado
# ============================================

output_path = Path(
    "app/data/processed/master_games_final.csv"
)

df.to_csv(output_path, index=False)

print("\n====================================")
print("Dataset actualizado guardado")
print(output_path)
print("====================================")
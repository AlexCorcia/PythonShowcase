from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt


# ============================================
# Cargar dataset
# ============================================

df = pd.read_csv(
    "app/data/processed/master_games_enriched.csv"
)

print("\nDataset cargado:")
print(df.shape)


# ============================================
# Features seleccionadas
# ============================================

features = [
    "num_moves",
    "player_captures",
    "player_checks",
    "aggression_score",
    "capture_rate",
    "check_rate",
]

target = "style"


# ============================================
# Eliminar nulos
# ============================================

df = df.dropna(subset=features + [target])

print("\nDataset tras limpiar nulos:")
print(df.shape)


# ============================================
# Variables X e y
# ============================================

X = df[features]
y = df[target]


# ============================================
# Split train/test
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)


# ============================================
# Modelo Random Forest
# ============================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
)

model.fit(X_train, y_train)


# ============================================
# Predicciones
# ============================================

y_pred = model.predict(X_test)


# ============================================
# Métricas
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\n======================================")
print("ACCURACY:")
print(round(accuracy, 4))
print("======================================")

print("\nCLASSIFICATION REPORT:\n")

print(classification_report(y_test, y_pred))


# ============================================
# Matriz de confusión
# ============================================

cm = confusion_matrix(y_test, y_pred)

print("\nCONFUSION MATRIX:\n")
print(cm)


# ============================================
# Feature importance
# ============================================

importance_df = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_,
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False,
)

print("\nFEATURE IMPORTANCE:\n")
print(importance_df)


# ============================================
# Gráfica importance
# ============================================

results_dir = Path("app/results/figures")
results_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 5))

plt.bar(
    importance_df["feature"],
    importance_df["importance"],
)

plt.title("Importancia de variables - Random Forest")

plt.xlabel("Feature")
plt.ylabel("Importance")

plt.xticks(rotation=20)

plt.tight_layout()

output_path = (
    results_dir /
    "random_forest_feature_importance.png"
)

plt.savefig(output_path)

plt.close()

print("\nGráfica guardada en:")
print(output_path)
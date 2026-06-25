from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier


# ============================================
# Cargar dataset temporal
# ============================================

df = pd.read_csv("app/data/processed/master_games_temporal.csv")

print("\nDataset temporal cargado:")
print(df.shape)


# ============================================
# Features temporales
# ============================================

temporal_features = [
    "opening_plies",
    "opening_player_captures",
    "opening_player_checks",
    "opening_player_castles",
    "opening_player_promotions",
    "opening_opponent_captures",
    "opening_opponent_checks",

    "middlegame_plies",
    "middlegame_player_captures",
    "middlegame_player_checks",
    "middlegame_player_castles",
    "middlegame_player_promotions",
    "middlegame_opponent_captures",
    "middlegame_opponent_checks",

    "endgame_plies",
    "endgame_player_captures",
    "endgame_player_checks",
    "endgame_player_castles",
    "endgame_player_promotions",
    "endgame_opponent_captures",
    "endgame_opponent_checks",
]

numeric_features = [
    "num_moves",
] + temporal_features

categorical_features = [
    "eco_family",
]

target = "style"


# ============================================
# Limpieza
# ============================================

df = df.dropna(subset=numeric_features + categorical_features + [target])

X = df[numeric_features + categorical_features]
y = df[target]


# ============================================
# Codificar labels para XGBoost
# ============================================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nClases:")
for idx, class_name in enumerate(label_encoder.classes_):
    print(idx, "->", class_name)


# ============================================
# Train / Test split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)


# ============================================
# Preprocessing
# ============================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)


# ============================================
# Modelos temporales
# ============================================

models = {
    "Temporal Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42,
    ),

    "Temporal XGBoost": XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.5,
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss",
        random_state=42,
    ),
}


# ============================================
# Carpetas resultados
# ============================================

results_dir = Path("app/results")
figures_dir = results_dir / "figures"
tables_dir = results_dir / "tables"

figures_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)


# ============================================
# Entrenamiento y evaluación
# ============================================

summary_rows = []

for model_name, model in models.items():

    print("\n======================================")
    print(f"Entrenando modelo: {model_name}")
    print("======================================")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    summary_rows.append(
        {
            "model": model_name,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }
    )

    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))

    # Classification report
    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
    )

    report_df = pd.DataFrame(report).transpose()

    safe_name = model_name.lower().replace(" ", "_")

    report_path = tables_dir / f"{safe_name}_classification_report.csv"
    report_df.to_csv(report_path)

    print("\nClassification report guardado:")
    print(report_path)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    )

    cm_path = tables_dir / f"{safe_name}_confusion_matrix.csv"
    cm_df.to_csv(cm_path)

    print("\nMatriz de confusión guardada:")
    print(cm_path)

    # Figura matriz de confusión
    plt.figure(figsize=(7, 6))
    plt.imshow(cm)

    plt.title(f"Matriz de confusión - {model_name}")
    plt.xlabel("Predicción")
    plt.ylabel("Clase real")

    plt.xticks(
        range(len(label_encoder.classes_)),
        label_encoder.classes_,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(label_encoder.classes_)),
        label_encoder.classes_,
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()

    cm_fig_path = figures_dir / f"{safe_name}_confusion_matrix.png"
    plt.savefig(cm_fig_path)
    plt.close()

    print("\nFigura matriz de confusión guardada:")
    print(cm_fig_path)


# ============================================
# Tabla resumen
# ============================================

summary_df = pd.DataFrame(summary_rows)

summary_path = tables_dir / "temporal_model_comparison_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("\n======================================")
print("Resumen comparativo temporal")
print("======================================")
print(summary_df.round(4))

print("\nTabla resumen guardada:")
print(summary_path)


# ============================================
# Figura comparación accuracy
# ============================================

plt.figure(figsize=(8, 5))
plt.bar(summary_df["model"], summary_df["accuracy"])

plt.title("Comparativa de accuracy - modelos temporales")
plt.xlabel("Modelo")
plt.ylabel("Accuracy")

plt.xticks(rotation=15, ha="right")
plt.tight_layout()

comparison_fig_path = figures_dir / "temporal_model_accuracy_comparison.png"
plt.savefig(comparison_fig_path)
plt.close()

print("\nFigura comparativa guardada:")
print(comparison_fig_path)
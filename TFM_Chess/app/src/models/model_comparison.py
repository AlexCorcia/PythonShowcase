from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================
# Cargar dataset
# ============================================

df = pd.read_csv(
    "app/data/processed/master_games_final.csv"
)

print("\nDataset cargado:")
print(df.shape)


# ============================================
# Features
# ============================================

numeric_features = [
    "num_moves",
    "player_captures",
    "player_checks",
    "aggression_score",
    "capture_rate",
    "check_rate",
]

categorical_features = [
    "eco_family",
]

target = "style"

df = df.dropna(
    subset=numeric_features + categorical_features + [target]
)

X = df[numeric_features + categorical_features]
y = df[target]


# ============================================
# Train / Test Split
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
# Preprocessing
# ============================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features,
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)


# ============================================
# Modelos
# ============================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
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

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
    )

    summary_rows.append(
        {
            "model": model_name,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }
    )

    print("\nAccuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))

    # ============================================
    # Classification report
    # ============================================

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
    )

    report_df = pd.DataFrame(report).transpose()

    report_path = (
        tables_dir /
        f"{model_name.lower().replace(' ', '_')}_classification_report.csv"
    )

    report_df.to_csv(report_path)

    print("\nClassification report guardado:")
    print(report_path)

    # ============================================
    # Confusion matrix
    # ============================================

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=sorted(y.unique()),
    )

    cm_df = pd.DataFrame(
        cm,
        index=sorted(y.unique()),
        columns=sorted(y.unique()),
    )

    cm_path = (
        tables_dir /
        f"{model_name.lower().replace(' ', '_')}_confusion_matrix.csv"
    )

    cm_df.to_csv(cm_path)

    print("\nMatriz de confusión guardada:")
    print(cm_path)

    # ============================================
    # Figura confusion matrix
    # ============================================

    plt.figure(figsize=(7, 6))

    plt.imshow(cm)

    plt.title(f"Matriz de confusión - {model_name}")

    plt.xlabel("Predicción")
    plt.ylabel("Clase real")

    plt.xticks(
        range(len(sorted(y.unique()))),
        sorted(y.unique()),
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(sorted(y.unique()))),
        sorted(y.unique()),
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    plt.tight_layout()

    cm_fig_path = (
        figures_dir /
        f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    )

    plt.savefig(cm_fig_path)

    plt.close()

    print("\nFigura confusion matrix guardada:")
    print(cm_fig_path)


# ============================================
# Tabla resumen comparativa
# ============================================

summary_df = pd.DataFrame(summary_rows)

summary_path = (
    tables_dir /
    "model_comparison_summary.csv"
)

summary_df.to_csv(summary_path, index=False)

print("\n======================================")
print("Resumen comparativo")
print("======================================")

print(summary_df.round(4))

print("\nTabla resumen guardada:")
print(summary_path)


# ============================================
# Figura accuracy comparación
# ============================================

plt.figure(figsize=(8, 5))

plt.bar(
    summary_df["model"],
    summary_df["accuracy"],
)

plt.title("Comparativa de accuracy entre modelos")

plt.xlabel("Modelo")
plt.ylabel("Accuracy")

plt.tight_layout()

comparison_fig_path = (
    figures_dir /
    "model_accuracy_comparison.png"
)

plt.savefig(comparison_fig_path)

plt.close()

print("\nFigura comparativa guardada:")
print(comparison_fig_path)
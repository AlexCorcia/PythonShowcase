from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier


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


# ============================================
# Limpiar nulos
# ============================================

df = df.dropna(
    subset=numeric_features + categorical_features + [target]
)

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
# Train / Test Split
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
# Modelo XGBoost
# ============================================

model = XGBClassifier(
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
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)


# ============================================
# Entrenamiento
# ============================================

print("\nEntrenando XGBoost...")

pipeline.fit(X_train, y_train)


# ============================================
# Predicción
# ============================================

y_pred = pipeline.predict(X_test)


# ============================================
# Métricas
# ============================================

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

print("\n======================================")
print("RESULTADOS XGBOOST")
print("======================================")

print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))
print("Weighted F1:", round(weighted_f1, 4))


# ============================================
# Carpetas resultados
# ============================================

results_dir = Path("app/results")

figures_dir = results_dir / "figures"
tables_dir = results_dir / "tables"

figures_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)


# ============================================
# Classification report
# ============================================

report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    output_dict=True,
)

report_df = pd.DataFrame(report).transpose()

report_path = tables_dir / "xgboost_classification_report.csv"

report_df.to_csv(report_path)

print("\nClassification report guardado:")
print(report_path)


# ============================================
# Confusion matrix
# ============================================

cm = confusion_matrix(
    y_test,
    y_pred,
)

cm_df = pd.DataFrame(
    cm,
    index=label_encoder.classes_,
    columns=label_encoder.classes_,
)

cm_path = tables_dir / "xgboost_confusion_matrix.csv"

cm_df.to_csv(cm_path)

print("\nMatriz de confusión guardada:")
print(cm_path)


# ============================================
# Figura matriz de confusión
# ============================================

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title("Matriz de confusión - XGBoost")

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
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
        )

plt.tight_layout()

cm_fig_path = figures_dir / "xgboost_confusion_matrix.png"

plt.savefig(cm_fig_path)

plt.close()

print("\nFigura matriz de confusión guardada:")
print(cm_fig_path)


# ============================================
# Feature names tras preprocessing
# ============================================

preprocessor_fitted = pipeline.named_steps["preprocessor"]

numeric_names = numeric_features

categorical_names = (
    preprocessor_fitted
    .named_transformers_["cat"]
    .get_feature_names_out(categorical_features)
    .tolist()
)

feature_names = numeric_names + categorical_names


# ============================================
# Feature importance
# ============================================

xgb_model = pipeline.named_steps["model"]

importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": xgb_model.feature_importances_,
    }
)

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False,
)

importance_path = tables_dir / "xgboost_feature_importance.csv"

importance_df.to_csv(
    importance_path,
    index=False,
)

print("\nFeature importance:")
print(importance_df.round(4))

print("\nFeature importance guardado:")
print(importance_path)


# ============================================
# Figura feature importance
# ============================================

plt.figure(figsize=(9, 5))

plt.bar(
    importance_df["feature"],
    importance_df["importance"],
)

plt.title("Importancia de variables - XGBoost")

plt.xlabel("Feature")
plt.ylabel("Importance")

plt.xticks(rotation=30, ha="right")

plt.tight_layout()

importance_fig_path = figures_dir / "xgboost_feature_importance.png"

plt.savefig(importance_fig_path)

plt.close()

print("\nFigura feature importance guardada:")
print(importance_fig_path)
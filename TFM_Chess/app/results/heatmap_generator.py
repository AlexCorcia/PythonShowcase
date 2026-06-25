import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Matriz de confusión
cm = np.array([
    [70, 21, 225, 59],
    [36, 46, 265, 78],
    [63, 45, 506, 90],
    [46, 56, 294, 88]
])

# Etiquetas
labels = ["Karpov", "Petrosian", "Kasparov", "Tal"]

# Crear heatmap
plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Heatmap")

plt.tight_layout()
plt.show()
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


LABEL_TO_STYLE = {
    0: "defensive",
    1: "dynamic",
    2: "positional",
    3: "tactical",
}


class GCNClassifier(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, num_classes)

        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.lin2(x)

        return x


def train_one_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        out = model(
            batch.x,
            batch.edge_index,
            batch.batch,
        )

        loss = F.cross_entropy(
            out,
            batch.y.view(-1),
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            out = model(
                batch.x,
                batch.edge_index,
                batch.batch,
            )

            pred = out.argmax(dim=1)

            y_true.extend(batch.y.view(-1).cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    return accuracy, macro_f1, weighted_f1, y_true, y_pred


def main():
    dataset_path = Path("app/data/graphs/graph_dataset.pt")

    results_dir = Path("app/results")
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    dataset = torch.load(dataset_path, weights_only=False)

    print("\nDataset GNN cargado:")
    print("Número de grafos:", len(dataset))
    print("Primer grafo:", dataset[0])
    print("Node features:", dataset[0].x.shape)
    print("Edges:", dataset[0].edge_index.shape)

    labels = [data.y.item() for data in dataset]

    train_data, test_data = train_test_split(
        dataset,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=32,
        shuffle=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    input_dim = dataset[0].x.shape[1]
    hidden_dim = 128
    num_classes = 4

    model = GCNClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0005,
        weight_decay=1e-4,
    )

    epochs = 150

    train_losses = []

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        train_losses.append(loss)

        if epoch % 5 == 0 or epoch == 1:
            accuracy, macro_f1, weighted_f1, _, _ = evaluate(
                model,
                test_loader,
                device,
            )

            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss:.4f} | "
                f"Acc: {accuracy:.4f} | "
                f"Macro F1: {macro_f1:.4f} | "
                f"Weighted F1: {weighted_f1:.4f}"
            )

    accuracy, macro_f1, weighted_f1, y_true, y_pred = evaluate(
        model,
        test_loader,
        device,
    )

    print("\n======================================")
    print("RESULTADOS GNN")
    print("======================================")
    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))

    target_names = [
        LABEL_TO_STYLE[i]
        for i in range(num_classes)
    ]

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
    )

    report_df = pd.DataFrame(report).transpose()

    report_path = tables_dir / "gnn_classification_report.csv"
    report_df.to_csv(report_path)

    print("\nClassification report guardado:")
    print(report_path)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
    )

    cm_df = pd.DataFrame(
        cm,
        index=target_names,
        columns=target_names,
    )

    cm_path = tables_dir / "gnn_confusion_matrix.csv"
    cm_df.to_csv(cm_path)

    print("\nMatriz de confusión guardada:")
    print(cm_path)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm)

    plt.title("Matriz de confusión - GNN")
    plt.xlabel("Predicción")
    plt.ylabel("Clase real")

    plt.xticks(
        range(num_classes),
        target_names,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(num_classes),
        target_names,
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

    cm_fig_path = figures_dir / "gnn_confusion_matrix.png"
    plt.savefig(cm_fig_path)
    plt.close()

    print("\nFigura matriz de confusión guardada:")
    print(cm_fig_path)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses)
    plt.title("Evolución de la pérdida de entrenamiento - GNN")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()

    loss_fig_path = figures_dir / "gnn_training_loss.png"
    plt.savefig(loss_fig_path)
    plt.close()

    print("\nFigura loss guardada:")
    print(loss_fig_path)

    model_path = results_dir / "gnn_model.pt"
    torch.save(model.state_dict(), model_path)

    print("\nModelo guardado:")
    print(model_path)


if __name__ == "__main__":
    main()
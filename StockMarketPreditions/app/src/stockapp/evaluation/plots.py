import numpy as np
import matplotlib.pyplot as plt


def plot_true_vs_pred(dates, y_true_1d, y_pred_1d, title: str):
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true_1d, label="True")
    plt.plot(dates, y_pred_1d, label="Predicted")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_true_vs_pred_multi(dates, series: list[tuple[str, np.ndarray]], title: str, y_label: str = "Price"):
    plt.figure(figsize=(14, 6))
    for label, y in series:
        plt.plot(dates, np.asarray(y).reshape(-1), label=label)
    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

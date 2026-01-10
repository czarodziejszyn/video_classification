import pandas as pd
import matplotlib.pyplot as plt

def plot_training_metrics(csv_path="logs/training_metrics.csv"):
    df = pd.read_csv(csv_path)

    epochs = df["epoch"]

    # ===== LOSS =====
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df["train_loss"], label="Train Loss")
    plt.plot(epochs, df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ===== ACCURACY =====
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df["train_acc"], label="Train Accuracy")
    plt.plot(epochs, df["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ===== LEARNING RATE =====
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df["lr"])
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_training_metrics()

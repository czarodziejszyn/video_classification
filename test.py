import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import NTUDataset
from src.models.st_transformer import STTransformer
from src.utils.paths import RAW_DATA_DIR

def evaluate_model(model, dataloader, device="cuda"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    print(f"Overall accuracy: {acc:.4f}")

    print("\nClassification report per class:")
    print(classification_report(all_labels, all_preds, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    return all_preds, all_labels, cm

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_set = NTUDataset(RAW_DATA_DIR, split="test")
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=4)


    model = STTransformer(num_classes=120)
    checkpoint = torch.load("checkpoints/last.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    # Ewaluacja
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()


import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import NTUDataset
from src.models.st_transformer import STTransformer
from src.utils.paths import RAW_DATA_DIR

def top_k_accuracy(outputs, labels, k=5):
    _, topk = outputs.topk(k, dim=1)
    correct = topk.eq(labels.view(-1, 1).expand_as(topk))
    return correct.any(dim=1).float().mean().item()


def evaluate_model(model, dataloader, device="cuda"):
    model.eval()
    all_preds = []
    all_labels = []

    top1_correct = 0
    top5_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            _, top5 = outputs.topk(5, dim=1)     # (B, 5)
            top1 = top5[:, 0]                     # najlepsza predykcja

            top1_correct += (top1 == y).sum().item()
            top5_correct += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
            total_samples += y.size(0)

            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    top1_acc = top1_correct / total_samples
    top5_acc = top5_correct / total_samples

    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")

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

    return {
        "top1": top1_acc,
        "top5": top5_acc,
        "accuracy": acc,
        "confusion_matrix": cm,
        "preds": all_preds,
        "labels": all_labels
    }

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


import torch
from src.utils.metrics import accuracy

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    val_acc1 = 0
    count = 0

    with torch.no_grad():
        for x, label in loader:
            x = x.to(device)
            label = label.to(device)

            out = model(x)
            loss = criterion(out, label)

            acc1 = accuracy(out, label, topk=(1,))[0]

            val_loss += loss.item() * x.size(0)
            val_acc1 += acc1 * x.size(0)
            count += x.size(0)

    return val_loss / count, val_acc1 / count

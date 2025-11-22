import os
import csv
import torch
from torch import nn
from tqdm import tqdm

from src.training.evaluate import validate
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.paths import (
    metrics_csv_path,
    checkpoint_path,
    last_checkpoint_path,
)

def train(model, train_loader, val_loader, optimizer, scheduler, cfg):
    device = cfg["device"]
    epochs = cfg["epochs"]
    resume = cfg.get("resume", False)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1

    # resume training
    if resume and os.path.exists(last_checkpoint_path()):
        print(f"Resuming from checkpoint: {last_checkpoint_path()}")
        start_epoch = load_checkpoint(
            model, optimizer, scheduler, last_checkpoint_path()
        ) + 1

    model.to(device)

    # metrics csv
    metrics_file = metrics_csv_path()
    write_header = not os.path.exists(metrics_file)

    csv_file = open(metrics_file, "a", newline="")
    csv_writer = csv.writer(csv_file)

    if write_header:
        csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    # TRAINING
    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        model.train()

        train_loss = 0
        train_acc = 0
        sample_count = 0

        for x, label in tqdm(train_loader, desc="Training"):
            x = x.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, label)

            loss.backward()
            optimizer.step()

            from src.utils.metrics import accuracy
            acc1 = accuracy(out, label, topk=(1,))[0]

            batch_size = x.size(0)
            train_loss += loss.item() * batch_size
            train_acc += acc1 * batch_size
            sample_count += batch_size

        train_loss /= sample_count
        train_acc /= sample_count

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        print(f"Train: loss={train_loss:.4f}, acc={train_acc:.2f}%")
        print(f"Val:   loss={val_loss:.4f}, acc={val_acc:.2f}%, lr={lr:.6f}")

        csv_writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, lr])
        csv_file.flush()

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            cfg=cfg,
            path=checkpoint_path(epoch),
        )
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            cfg=cfg,
            path=last_checkpoint_path(),
        )

        print(f"Checkpoint saved: epoch_{epoch:03d}.pt")

    csv_file.close()

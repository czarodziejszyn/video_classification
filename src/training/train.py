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
    warmup_epochs = cfg.get("warmup_epochs", 5)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    start_epoch = 1

    model.to(device)

    # resume training
    if resume and os.path.exists(last_checkpoint_path()):
        print(f"Resuming from checkpoint: {last_checkpoint_path()}")
        start_epoch = load_checkpoint(
            model, optimizer, scheduler, scaler, last_checkpoint_path()
        )
        start_epoch += 1



    # metrics csv
    metrics_file = metrics_csv_path()
    write_header = not os.path.exists(metrics_file)

    csv_file = open(metrics_file, "a", newline="")
    csv_writer = csv.writer(csv_file)

    if write_header:
        csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    # TRAINING LOOP
    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        model.train()

        train_loss = 0
        train_acc = 0
        sample_count = 0

        pbar = tqdm(train_loader, desc="Training")

        for x, label in pbar:
            x = x.to(device)
            label = label.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                out = model(x)
                loss = criterion(out, label)

            # AMP backward + optimizer
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            from src.utils.metrics import accuracy
            acc1 = accuracy(out, label, topk=(1,))[0]

            batch_size = x.size(0)

            train_loss += loss.item() * batch_size
            train_acc += acc1 * batch_size
            sample_count += batch_size

        train_loss /= sample_count
        train_acc /= sample_count

        # -------- VALIDATION --------
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        print(f"Train: loss={train_loss:.4f}, acc={train_acc:.2f}%")
        print(f"Val:   loss={val_loss:.4f}, acc={val_acc:.2f}%, lr={lr:.6f}")

        # Save CSV
        csv_writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, lr])
        csv_file.flush()

        # save checkpoints
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, cfg, last_checkpoint_path())
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, cfg, checkpoint_path(epoch))

    csv_file.close()

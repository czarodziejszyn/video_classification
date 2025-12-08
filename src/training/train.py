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

    start_epoch = 1
    best_acc = 0.0

    # resume training
    if resume and os.path.exists(last_checkpoint_path()):
        print(f"Resuming from checkpoint: {last_checkpoint_path()}")
        start_epoch, best_acc = load_checkpoint(
            model, optimizer, scheduler, last_checkpoint_path()
        )
        start_epoch += 1

    model.to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

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

        for x, label in tqdm(train_loader, desc=f"Training"):
            x = x.to(device)
            label = label.to(device)

            optimizer.zero_grad(set_to_none=True)

            # mixed precision
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                out = model(x)
                loss = criterion(out, label)

            scaler.scale(loss).backward()

            # gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            # accuracy
            pred = out.argmax(dim=1)
            correct = (pred == label).sum().item()

            batch_size = x.size(0)
            train_loss += loss.item() * batch_size
            train_acc += correct
            sample_count += batch_size

        train_loss /= sample_count
        train_acc = 100.0 * train_acc / sample_count

        # validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # warm-up
        if epoch <= warmup_epochs:
            lr_scale = epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = cfg["lr"] * lr_scale
            lr = optimizer.param_groups[0]["lr"]
        else:
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

        print(f"Train: loss={train_loss:.4f}, acc={train_acc:.2f}%")
        print(f"Val:   loss={val_loss:.4f}, acc={val_acc:.2f}%, lr={lr:.6f}")

        csv_writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, lr])
        csv_file.flush()

        # save checkpoints
        save_checkpoint(model, optimizer, scheduler, epoch, cfg, last_checkpoint_path())
        save_checkpoint(model, optimizer, scheduler, epoch, cfg, checkpoint_path(epoch))

    csv_file.close()

import torch
from src.utils.paths import last_checkpoint_path

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, cfg, path):
    ckpt= {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": cfg,
    }
    torch.save(ckpt, path)

def load_checkpoint(model, optimizer, scheduler, scaler, path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])

    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    return ckpt["epoch"]

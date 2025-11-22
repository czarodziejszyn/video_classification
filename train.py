import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.dataset import NTUDataset
from src.models.st_transformer import STTransformer
from src.training.train import train
from src.utils.paths import RAW_DATA_DIR


def main():

    train_set = NTUDataset(RAW_DATA_DIR, split="train")
    val_set = NTUDataset(RAW_DATA_DIR, split="val")

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=4)

    model = STTransformer(num_classes=120)

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    cfg = {
        "epochs": 50,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "resume": True,
    }

    train(model, train_loader, val_loader, optimizer, scheduler, cfg)


if __name__ == "__main__":
    main()

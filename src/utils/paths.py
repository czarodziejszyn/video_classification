import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

LOGS_DIR = PROJECT_ROOT / "logs"

def ensure_dirs():
    """Creates directories if they don't exist."""
    for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)

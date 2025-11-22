import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from .parsing import load_skeleton
from .labels import get_label_from_filename
from src.utils.paths import PROJECT_ROOT


def select_main_person(persons):
    """Selects the person with the highest motion score."""
    if len(persons) == 0:
        return None

    if len(persons) == 1:
        return persons[0]

    motion_scores = []
    for p in persons:
        diffs = np.diff(p, axis=0)
        motion = np.sum(np.linalg.norm(diffs, axis=2))
        motion_scores.append(motion)

    idx = int(np.argmax(motion_scores))
    return persons[idx]


def normalize_skeleton(seq):
    center = seq[:, 0:1, :]
    return seq - center


def resize_seq(seq, target_len=100):
    T = seq.shape[0]
    if T == target_len:
        return seq

    idxs = np.linspace(0, T - 1, target_len).astype(np.int32)
    return seq[idxs]


class NTUDataset(Dataset):
    """
    Dataset for NTU RGB+D skeletons.
    """

    def __init__(self, skeleton_dir, split="train", file_list=None, target_len=100, transform=None):

        self.skeleton_dir = Path(skeleton_dir)

        # Load excluded files
        excluded_path = Path(PROJECT_ROOT) / "src" / "data" / "excluded_files.txt"
        if excluded_path.exists():
            with open(excluded_path, "r") as f:
                excluded_files = set(line.strip() for line in f if line.strip())
        else:
            excluded_files = set()

        all_files = [
            f for f in sorted(self.skeleton_dir.glob("*.skeleton"))
            if f.name not in excluded_files
        ]

        # Proper split logic (not overwritten later)
        if file_list is not None:
            self.files = [self.skeleton_dir / f for f in file_list]

        else:
            N = len(all_files)
            if split == "train":
                self.files = all_files[: int(N * 0.8)]
            elif split == "val":
                self.files = all_files[int(N * 0.8): int(N * 0.9)]
            elif split == "test":
                self.files = all_files[int(N * 0.9):]
            elif split == "all":
                self.files = all_files
            else:
                raise ValueError(f"Unknown split: {split}")

        # Save path for logging bad files
        self.bad_file_log = Path(PROJECT_ROOT) / "bad_samples.txt"

        self.target_len = target_len
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        try:
            parsed = load_skeleton(path)

            persons = parsed.get("persons", [])
            main = select_main_person(persons)

            if main is None:
                raise ValueError("No valid persons in file")

            main = normalize_skeleton(main)
            main = resize_seq(main)

            if self.transform is not None:
                main = self.transform(main)

            main = torch.tensor(main, dtype=torch.float32)
            label = get_label_from_filename(path)

            return main, label

        except Exception as e:

            # Log to file, not stdout
            with open(self.bad_file_log, "a") as f:
                f.write(f"{path} | ERROR: {str(e)}\n")

            # Return dummy sample so DataLoader doesn't crash
            dummy = torch.zeros(self.target_len, 25, 3, dtype=torch.float32)
            label = 0  # or -1 if you prefer to ignore them later

            return dummy, label


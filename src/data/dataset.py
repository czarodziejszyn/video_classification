import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import re

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
    """Center skeleton around the first joint."""
    center = seq[:, 0:1, :]
    return seq - center


def resize_seq(seq, target_len=100):
    """Resize or sample sequence to fixed length."""
    T = seq.shape[0]
    if T == target_len:
        return seq
    idxs = np.linspace(0, T - 1, target_len).astype(np.int32)
    return seq[idxs]


def get_subject_from_filename(filename):
    """
    Extract person ID from NTU skeleton filename.
    Example: 'S001C001P001R001.skeleton' -> 1
    """
    match = re.search(r'P(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Cannot parse subject from filename: {filename}")

class NTUDataset(Dataset):
    """
    Dataset for NTU RGB+D 120 skeletons with official cross-subject split.
    """
    
    # Oficjalne ID aktorów do zbioru TRENINGOWEGO dla NTU 120
    TRAIN_IDS = {
        1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38,
        45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 
        82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
    }

    def __init__(self, skeleton_dir, split="train", target_len=100, transform=None):
        self.skeleton_dir = Path(skeleton_dir)
        self.target_len = target_len
        self.transform = transform
        self.split = split

        # Wczytanie plików do wykluczenia (tzw. missing skeletons)
        excluded_path = Path(PROJECT_ROOT) / "src" / "data" / "excluded_files.txt"
        excluded_files = set()
        if excluded_path.exists():
            with open(excluded_path, "r") as f:
                excluded_files = set(line.strip() for line in f if line.strip())

        # Pobranie i sortowanie wszystkich plików .skeleton
        all_files = sorted(self.skeleton_dir.glob("*.skeleton"))
        self.files = []

        for f in all_files:
            # Pomiń pliki z listy uszkodzonych
            if f.name in excluded_files:
                continue
                
            try:
                # Wyciągnięcie ID aktora z nazwy (np. P003 -> 3)
                subject = get_subject_from_filename(f.name)
                is_train_subject = subject in self.TRAIN_IDS
                
                # Przypisanie do odpowiedniego splitu
                if split == "train" and is_train_subject:
                    self.files.append(f)
                elif split == "test" and not is_train_subject:
                    self.files.append(f)
                elif split == "all":
                    self.files.append(f)
                    
            except Exception as e:
                print(f"Skipping file {f.name}: {e}")

        self.bad_file_log = Path(PROJECT_ROOT) / "bad_samples.txt"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        try:
            parsed = load_skeleton(path)
            persons = parsed.get("persons", [])
            main = select_main_person(persons)

            if main is None:
                raise ValueError(f"No valid persons in file: {path.name}")

            # Preprocessing
            main = normalize_skeleton(main)
            main = resize_seq(main, self.target_len)

            if self.transform is not None:
                main = self.transform(main)

            # Konwersja na tensor
            main = torch.tensor(main, dtype=torch.float32)
            label = get_label_from_filename(path.name)

            return main, label

        except Exception as e:
            # Logowanie błędnych plików
            with open(self.bad_file_log, "a") as f:
                f.write(f"{path.name} | ERROR: {str(e)}\n")

            # Zwracanie dummy sample z label -1 (do odfiltrowania w DataLoaderze)
            dummy = torch.zeros((self.target_len, 25, 3), dtype=torch.float32)
            return dummy, -1


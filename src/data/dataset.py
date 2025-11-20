import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from .parsing import load_skeleton
from .labels import get_label_from_filename

def select_main_person(persons):
    """Selects person that moves the most (highest sum of distances between frames)"""
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
    """
    Normalize skeleton on every frame - center for hip joint.

    seq: numpy array [T x 25 x 3]
    """
    center = seq[:, 0:1, :]
    return seq - center

def resize_seq(seq, target_len=100):
    """
    Resize every sequence to 100 frames.
    """
    T = seq.shape[0]

    if T == target_len:
        return seq

    idxs = np.linspace(0, T - 1, target_len).astype(np.int32)
    return seq[idxs]

class NTUDataset(Dataset):
    """
    Dataset for NTU RGB+D skeletons.
    """
    def __init__(self, skeleton_dir, file_list=None, target_len=100, transform=None):
        """
        skeleton_dir: directory with *.skeleton files
        file_list: names of files or None to use all
        target_len: sequence length after interpolation
        transform: augmentations
        """
        self.skeleton_dir = Path(skeleton_dir)

        if file_list is None:
            self.files = sorted(self.skeleton_dir.glob("*.skeleton"))
        else:
            self.files = [self.skeleton_dir / f for f in file_list]

        self.target_len = target_len
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        parsed = load_skeleton(path)

        person = select_main_person(parsed["persons"])
        person = normalize_skeleton(person)
        person = resize_seq(person)

        if self.transform is not None:
            person = self.transform(person)

        person = torch.tensor(person, dtype=torch.float32)

        label = get_label_from_filename(path)

        return person, label

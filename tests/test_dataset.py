import os
import pytest
import numpy as np

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.dataset import NTUDataset
from src.utils.paths import RAW_DATA_DIR


def test_dataset_initialization():
    """
    Tests dataset initialization.
    """
    dataset = NTUDataset(RAW_DATA_DIR)
    assert len(dataset) > 0

def test_dataset_first_sample():
    """
    Tests loading of first sample from dataset.
    """
    dataset = NTUDataset(RAW_DATA_DIR)
    sample = dataset[0]
    print(sample)

    assert len(sample) == 2

    x = sample[0]
    label = sample[1]

    assert x.ndim == 3

    T, V, C = x.shape

    assert T == 100
    assert V == 25
    assert C == 3

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Sets seed for all libraries for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    """Seed for DataLoader workers."""
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)

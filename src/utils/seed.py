"""Reproducibility helpers used across every training/eval script."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix every RNG we touch (Python, NumPy, PyTorch, CUDA) for reproducible runs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Trade a little speed for determinism; flip off if you need max throughput.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

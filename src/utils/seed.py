"""Reproducibility and RNG seeding utilities."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Sets RNG seed across random, numpy, and PyTorch for deterministic reproducibility.
    """
    random.seed(seed)
    # Propagate seed to child subprocesses (interpreter hash seed is set before process start)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

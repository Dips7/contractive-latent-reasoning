"""Symmetric group S_n word composition generator."""

import torch
import numpy as np
from typing import Tuple


def generate_permutation_dataset(
    num_samples: int = 10000,
    group_n: int = 5,
    composition_len: int = 16,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate sequences of permutations g_1, ..., g_L in S_n and their composition.
    
    Args:
        num_samples: Number of sequences
        group_n: Degree of symmetric group S_n
        composition_len: Sequence length L
        seed: Random seed
        
    Returns:
        words: LongTensor of shape (num_samples, composition_len, group_n)
        targets: LongTensor of shape (num_samples, group_n)
    """
    rng = np.random.default_rng(seed)
    words = []
    targets = []
    
    identity = np.arange(group_n)
    
    for _ in range(num_samples):
        current = identity.copy()
        word = []
        for _ in range(composition_len):
            # Sample random permutation in S_n
            p = rng.permutation(group_n)
            word.append(p)
            # Compose: current = p(current)
            current = p[current]
            
        words.append(word)
        targets.append(current)
        
    return torch.tensor(np.array(words), dtype=torch.long), torch.tensor(np.array(targets), dtype=torch.long)

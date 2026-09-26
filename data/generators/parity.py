"""N-bit parity dataset generator."""

import torch
from typing import Tuple


def generate_parity_dataset(
    num_samples: int = 10000,
    seq_len: int = 64,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate binary parity sequences where x in {-1, +1}^N, y = prod(x_i).
    
    Args:
        num_samples: Number of sequences to generate.
        seq_len: Length N of the binary sequence.
        seed: Random seed for reproducibility.
        
    Returns:
        x: FloatTensor of shape (num_samples, seq_len) in {-1.0, 1.0}
        y: LongTensor of shape (num_samples,) in {0, 1} where 1 represents parity +1
    """
    generator = torch.Generator().manual_seed(seed)
    # Generate Bernoulli {0, 1}
    raw_bits = torch.randint(0, 2, (num_samples, seq_len), generator=generator)
    # Map to {-1.0, 1.0}
    x = raw_bits.float() * 2.0 - 1.0
    # True parity in {-1, +1}
    prod = torch.prod(x, dim=1)
    # Map to classification target {0, 1}: +1 -> 1, -1 -> 0
    y = ((prod + 1.0) / 2.0).long()
    return x, y

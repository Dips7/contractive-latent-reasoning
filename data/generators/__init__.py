"""Synthetic dataset generators for algorithmic invariants."""

from .parity import generate_parity_dataset
from .graph_connectivity import generate_graph_dataset
from .permutation_groups import generate_permutation_dataset

__all__ = [
    "generate_parity_dataset",
    "generate_graph_dataset",
    "generate_permutation_dataset",
]

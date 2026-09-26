"""Directed graph connectivity dataset generator."""

import torch
import numpy as np
from typing import Tuple, Dict, Any


def generate_graph_dataset(
    num_samples: int = 10000,
    num_nodes: int = 32,
    edge_prob: float = 0.15,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Generate random directed Erdős–Rényi graphs and query reachability between source and target.
    
    Returns:
        Dictionary containing:
            - 'adj': Adjacency matrices (num_samples, num_nodes, num_nodes)
            - 'source': Source node indices (num_samples,)
            - 'target': Target node indices (num_samples,)
            - 'reachable': Reachability label (num_samples,) in {0, 1}
    """
    rng = np.random.default_rng(seed)
    adj_list = []
    sources = []
    targets = []
    labels = []
    
    for _ in range(num_samples):
        # Generate random adjacency matrix
        A = (rng.random((num_nodes, num_nodes)) < edge_prob).astype(np.float32)
        np.fill_diagonal(A, 0.0) # No self-loops
        
        # Pick random source and target
        u, v = rng.choice(num_nodes, size=2, replace=False)
        
        # Check reachability via matrix power / BFS
        # Transitive closure
        reachable = False
        visited = set([u])
        queue = [u]
        while queue:
            curr = queue.pop(0)
            if curr == v:
                reachable = True
                break
            neighbors = np.where(A[curr] > 0)[0]
            for nbr in neighbors:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
                    
        adj_list.append(A)
        sources.append(u)
        targets.append(v)
        labels.append(1 if reachable else 0)
        
    return {
        "adj": torch.tensor(np.array(adj_list), dtype=torch.float32),
        "source": torch.tensor(sources, dtype=torch.long),
        "target": torch.tensor(targets, dtype=torch.long),
        "reachable": torch.tensor(labels, dtype=torch.long),
    }

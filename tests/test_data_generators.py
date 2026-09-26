"""Tests for synthetic data generators."""

import pytest
import torch
from data.generators.parity import generate_parity_dataset
from data.generators.graph_connectivity import generate_graph_dataset
from data.generators.permutation_groups import generate_permutation_dataset


def test_parity_generator():
    seq_len = 16
    num_samples = 50
    x, y = generate_parity_dataset(num_samples=num_samples, seq_len=seq_len, seed=123)
    
    assert x.shape == (num_samples, seq_len)
    assert y.shape == (num_samples,)
    # Parity should strictly match product of x
    computed_prod = torch.prod(x, dim=1)
    expected_y = ((computed_prod + 1.0) / 2.0).long()
    assert torch.equal(y, expected_y)


def test_graph_generator():
    num_samples = 20
    num_nodes = 10
    data = generate_graph_dataset(num_samples=num_samples, num_nodes=num_nodes, edge_prob=0.2, seed=42)
    assert data["adj"].shape == (num_samples, num_nodes, num_nodes)
    assert data["source"].shape == (num_samples,)
    assert data["target"].shape == (num_samples,)
    assert data["reachable"].shape == (num_samples,)

    # Verify label semantics by independent BFS
    for i in range(num_samples):
        A = data["adj"][i].numpy()
        u = data["source"][i].item()
        v = data["target"][i].item()
        
        visited = set([u])
        queue = [u]
        is_reachable = False
        while queue:
            curr = queue.pop(0)
            if curr == v:
                is_reachable = True
                break
            for nxt in range(num_nodes):
                if A[curr, nxt] > 0 and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
                    
        assert data["reachable"][i].item() == (1 if is_reachable else 0)


def test_permutation_generator():
    num_samples = 15
    group_n = 4
    comp_len = 8
    words, targets = generate_permutation_dataset(
        num_samples=num_samples, group_n=group_n, composition_len=comp_len, seed=42
    )
    assert words.shape == (num_samples, comp_len, group_n)
    assert targets.shape == (num_samples, group_n)

    # Verify label semantics by independent sequential permutation composition
    for i in range(num_samples):
        curr = list(range(group_n))
        for step in range(comp_len):
            p = words[i, step].tolist()
            curr = [p[idx] for idx in curr]
        assert curr == targets[i].tolist()

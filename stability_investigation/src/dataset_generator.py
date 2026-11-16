"""
Synthetic Benchmark Generator for Controlled Graph Reachability.
Enforces:
1. Strict graph-level splits (Train, Val, Test on distinct graph instances).
2. Zero replacement sampling (no artificial duplication in sparse hop buckets).
3. Zero internal duplicate queries.
4. Mathematical verification of reachability and shortest-path distance for every query.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import networkx as nx
import numpy as np
import torch

def build_normalized_operator(N: int, row_indices: List[int], col_indices: List[int]) -> torch.Tensor:
    """
    Constructs normalized directed graph operator P = N^T with ||P||_2 <= 1.0.
    row is sender u, col is receiver v.
    Forward message passing along u -> v: (P z)_v = sum_{u} N_{u, v} z_u.
    """
    if len(row_indices) == 0:
        return torch.sparse_coo_tensor((2, 0), torch.tensor([], dtype=torch.float32), (N, N)).coalesce()
        
    row_t = torch.tensor(row_indices, dtype=torch.long)
    col_t = torch.tensor(col_indices, dtype=torch.long)
    
    d_out = torch.zeros(N)
    d_in = torch.zeros(N)
    for u, v in zip(row_indices, col_indices):
        d_out[u] += 1.0
        d_in[v] += 1.0
        
    d_out_inv_sqrt = torch.pow(d_out.clamp(min=1.0), -0.5)
    d_in_inv_sqrt = torch.pow(d_in.clamp(min=1.0), -0.5)
    
    values = d_out_inv_sqrt[row_t] * d_in_inv_sqrt[col_t]
    
    # Receiver is row in P, sender is col in P
    indices_P = torch.stack([col_t, row_t])
    P = torch.sparse_coo_tensor(indices_P, values, (N, N)).coalesce()
    return P


def generate_chain_ladder(length: int = 40, cross_prob: float = 0.3, seed: int = 42) -> Tuple[nx.DiGraph, torch.Tensor]:
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(length))
    row_indices, col_indices = [], []
    
    for i in range(length - 1):
        G.add_edge(i, i + 1)
        row_indices.append(i)
        col_indices.append(i + 1)
        
    for i in range(length - 2):
        if rng.random() < cross_prob:
            G.add_edge(i, i + 2)
            row_indices.append(i)
            col_indices.append(i + 2)
            
    P = build_normalized_operator(length, row_indices, col_indices)
    return G, P


def generate_bottleneck(cluster_size: int = 24, edge_prob: float = 0.20, seed: int = 42) -> Tuple[nx.DiGraph, torch.Tensor]:
    rng = np.random.default_rng(seed)
    N = 2 * cluster_size
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    row_indices, col_indices = [], []
    
    # Cluster A (DAG ordering to avoid cycles and maintain clean multi-hop paths)
    for i in range(cluster_size):
        for j in range(i + 1, cluster_size):
            if rng.random() < edge_prob:
                G.add_edge(i, j)
                row_indices.append(i)
                col_indices.append(j)
                
    # Cluster B (DAG ordering)
    for i in range(cluster_size, N):
        for j in range(i + 1, N):
            if rng.random() < edge_prob:
                G.add_edge(i, j)
                row_indices.append(i)
                col_indices.append(j)
                
    # Ensure connectivity within clusters if sparse
    for i in range(cluster_size - 1):
        if not G.has_edge(i, i + 1):
            G.add_edge(i, i + 1)
            row_indices.append(i)
            col_indices.append(i + 1)
    for i in range(cluster_size, N - 1):
        if not G.has_edge(i, i + 1):
            G.add_edge(i, i + 1)
            row_indices.append(i)
            col_indices.append(i + 1)
            
    # Bridge edge A -> B
    bridge_u = cluster_size - 1
    bridge_v = cluster_size
    G.add_edge(bridge_u, bridge_v)
    row_indices.append(bridge_u)
    col_indices.append(bridge_v)
    
    P = build_normalized_operator(N, row_indices, col_indices)
    return G, P


def generate_random_dag(num_layers: int = 7, nodes_per_layer: int = 6, edge_prob: float = 0.25, seed: int = 42) -> Tuple[nx.DiGraph, torch.Tensor]:
    rng = np.random.default_rng(seed)
    N = num_layers * nodes_per_layer
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    row_indices, col_indices = [], []
    
    # Ensure backbone connectivity between adjacent layers
    for l in range(num_layers - 1):
        u = l * nodes_per_layer + rng.integers(0, nodes_per_layer)
        v = (l + 1) * nodes_per_layer + rng.integers(0, nodes_per_layer)
        G.add_edge(u, v)
        row_indices.append(u)
        col_indices.append(v)
        
    # Additional forward edges
    for l1 in range(num_layers):
        for l2 in range(l1 + 1, num_layers):
            for u in range(l1 * nodes_per_layer, (l1 + 1) * nodes_per_layer):
                for v in range(l2 * nodes_per_layer, (l2 + 1) * nodes_per_layer):
                    if not G.has_edge(u, v) and rng.random() < edge_prob:
                        G.add_edge(u, v)
                        row_indices.append(u)
                        col_indices.append(v)
                        
    P = build_normalized_operator(N, row_indices, col_indices)
    return G, P


def sample_graph_queries(G: nx.DiGraph, seed: int, target_per_hop: int = 20, max_hops: int = 6) -> List[Dict[str, int]]:
    """
    Samples strictly non-overlapping, non-duplicate query pairs from G without replacement.
    Positives: stratified across hops 1..max_hops.
    Negatives: verified unreachable pairs.
    """
    rng = np.random.default_rng(seed)
    all_lengths = dict(nx.all_pairs_shortest_path_length(G))
    
    # Bin positive pairs by exact shortest-path distance
    hop_buckets: Dict[int, List[Tuple[int, int]]] = {h: [] for h in range(1, max_hops + 1)}
    for u, targets in all_lengths.items():
        for v, d in targets.items():
            if u != v and 1 <= d <= max_hops:
                hop_buckets[d].append((u, v))
                
    sampled_pos: List[Dict[str, int]] = []
    seen_pairs = set()
    
    for h in range(1, max_hops + 1):
        available = hop_buckets[h]
        if not available:
            continue
        n_take = min(target_per_hop, len(available))
        # STRICT NO-REPLACEMENT SAMPLING
        perm = rng.permutation(len(available))
        for idx in perm[:n_take]:
            u, v = available[idx]
            assert (u, v) not in seen_pairs, f"Duplicate positive pair detected: {(u, v)}"
            seen_pairs.add((u, v))
            sampled_pos.append({"source": int(u), "target": int(v), "label": 1, "hop": int(h)})
            
    # Sample negative pairs (unreachable)
    # Collect candidate negative pairs: out(u) > 0 and in(v) > 0 and no path
    out_nodes = [n for n in G.nodes() if G.out_degree(n) > 0]
    in_nodes = [n for n in G.nodes() if G.in_degree(n) > 0]
    
    all_neg_candidates = []
    for u in out_nodes:
        for v in in_nodes:
            if u != v and v not in all_lengths.get(u, {}):
                all_neg_candidates.append((u, v))
                
    target_neg = len(sampled_pos)
    assert len(all_neg_candidates) >= target_neg, f"Insufficient negative candidates: {len(all_neg_candidates)} < {target_neg}"
    
    perm_neg = rng.permutation(len(all_neg_candidates))
    sampled_neg: List[Dict[str, int]] = []
    for idx in perm_neg[:target_neg]:
        u, v = all_neg_candidates[idx]
        assert (u, v) not in seen_pairs, f"Duplicate or positive-overlapping negative pair detected: {(u, v)}"
        seen_pairs.add((u, v))
        sampled_neg.append({"source": int(u), "target": int(v), "label": 0, "hop": 0})
        
    all_queries = sampled_pos + sampled_neg
    rng.shuffle(all_queries)
    return all_queries


def generate_benchmark_suite(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    families = {
        "chain_ladder": {
            "gen_fn": lambda s: generate_chain_ladder(length=40, cross_prob=0.3, seed=s),
            "train_seeds": [101, 102, 103, 104, 105],
            "val_seeds": [201, 202],
            "test_seeds": [301, 302, 303, 304, 305],
            "target_per_hop": 20
        },
        "bottleneck": {
            "gen_fn": lambda s: generate_bottleneck(cluster_size=24, edge_prob=0.20, seed=s),
            "train_seeds": [111, 112, 113, 114, 115],
            "val_seeds": [211, 212],
            "test_seeds": [311, 312, 313, 314, 315],
            "target_per_hop": 20
        },
        "random_dag": {
            "gen_fn": lambda s: generate_random_dag(num_layers=7, nodes_per_layer=6, edge_prob=0.25, seed=s),
            "train_seeds": [121, 122, 123, 124, 125],
            "val_seeds": [221, 222],
            "test_seeds": [321, 322, 323, 324, 325],
            "target_per_hop": 20
        }
    }
    
    manifest: Dict[str, Any] = {
        "benchmark_name": "Controlled Synthetic Reachability Suite (Phase 2)",
        "protocol": "Strict Graph-Level Split with Zero Replacement Sampling",
        "families": {}
    }
    
    for fam_name, config in families.items():
        fam_dir = output_dir / fam_name
        fam_dir.mkdir(parents=True, exist_ok=True)
        manifest["families"][fam_name] = {"splits": {}}
        
        split_map = {
            "train": config["train_seeds"],
            "val": config["val_seeds"],
            "test": config["test_seeds"]
        }
        
        for split_name, seed_list in split_map.items():
            manifest["families"][fam_name]["splits"][split_name] = []
            for idx, seed in enumerate(seed_list):
                G, P = config["gen_fn"](seed)
                queries = sample_graph_queries(G, seed=seed * 10 + 7, target_per_hop=config["target_per_hop"])
                
                # Compute hop distribution
                hop_dist: Dict[int, int] = {}
                for q in queries:
                    h = q["hop"]
                    hop_dist[h] = hop_dist.get(h, 0) + 1
                    
                edge_list = list(G.edges())
                row_indices = [u for u, v in edge_list]
                col_indices = [v for u, v in edge_list]
                
                record = {
                    "family": fam_name,
                    "split": split_name,
                    "instance_idx": idx,
                    "seed": seed,
                    "num_nodes": G.number_of_nodes(),
                    "num_edges": G.number_of_edges(),
                    "edge_index": torch.tensor([row_indices, col_indices], dtype=torch.long),
                    "operator_P": P,
                    "queries": queries,
                    "hop_distribution": hop_dist
                }
                
                file_name = f"{split_name}_inst_{idx}_seed_{seed}.pt"
                file_path = fam_dir / file_name
                torch.save(record, file_path)
                
                meta_entry = {
                    "file_name": file_name,
                    "instance_idx": idx,
                    "seed": seed,
                    "num_nodes": G.number_of_nodes(),
                    "num_edges": G.number_of_edges(),
                    "total_queries": len(queries),
                    "hop_distribution": hop_dist
                }
                manifest["families"][fam_name]["splits"][split_name].append(meta_entry)
                
    manifest_path = output_dir / "benchmark_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated synthetic benchmark suite at {output_dir}")
    print(f"Manifest written to {manifest_path}")

if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "synthetic"
    generate_benchmark_suite(out)

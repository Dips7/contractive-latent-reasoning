# Synthetic Benchmark Data Generators

This directory hosts the procedural data generation pipelines designed to expose the depth-limited breakdown of standard discrete transformers without Chain-of-Thought (CoT).

## 1. N-Bit Parity (`generators/parity.py`)
- **Task**: Compute $y = \prod_{i=1}^N x_i \in \{-1, +1\}$ given $x_i \in \{-1, +1\}$.
- **Theoretical Complexity**: Requires computing parity across $N$ elements; fixed-depth circuits fail when $N \gg \text{depth}$.
- **Usage**:
  ```python
  from data.generators.parity import generate_parity_dataset
  x, y = generate_parity_dataset(num_samples=10000, seq_len=64)
  ```

## 2. Graph Connectivity (`generators/graph_connectivity.py`)
- **Task**: Determine if there exists a directed path between source node $u$ and target node $v$ within $k$ hops.
- **Usage**:
  ```python
  from data.generators.graph_connectivity import generate_graph_dataset
  graphs, queries, labels = generate_graph_dataset(num_samples=10000, num_nodes=32, k_hops=8)
  ```

## 3. Permutation Groups $S_n$ (`generators/permutation_groups.py`)
- **Task**: Given a word of permutations $g_1 \circ g_2 \circ \dots \circ g_L \in S_5$, compute the composite output permutation.
- **Theoretical Complexity**: Non-abelian state-tracking requiring dynamic compositional depth.

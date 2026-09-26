"""Automated tests for Real-World Cora Citation Reachability Pipeline."""

import pytest
import torch
from pathlib import Path

from data.generators.cora_reachability import load_cora_graph, generate_cora_reachability_splits
from src.models.cora_reasoner import (
    CoraDirectBaseline,
    CoraDiscreteGNN,
    CoraContractiveReasoner,
    compute_empirical_cora_sym_j,
)


@pytest.fixture(scope="module")
def cora_data():
    cora_dir = Path(__file__).resolve().parents[1] / "data" / "real_world" / "cora" / "cora"
    if not cora_dir.exists():
        pytest.skip("Cora raw files not found; skipping real-world data test.")
    G, A_sparse, paper_ids, id2idx = load_cora_graph(str(cora_dir))
    return G, A_sparse, paper_ids, id2idx


def test_cora_graph_loading(cora_data):
    G, A_sparse, paper_ids, id2idx = cora_data
    assert len(paper_ids) == 2708, f"Expected 2708 papers, got {len(paper_ids)}"
    assert G.number_of_nodes() == 2708
    assert G.number_of_edges() == 5429, f"Expected 5429 edges, got {G.number_of_edges()}"
    assert A_sparse.shape == (2708, 2708)
    assert A_sparse.is_sparse


def test_cora_spectral_norm(cora_data):
    """Converged power iteration (500 steps) verifying ||A_norm||_2 <= 1.000001 (C-4, C-5)."""
    _, A_sparse, paper_ids, _ = cora_data
    num_nodes = len(paper_ids)
    
    torch.manual_seed(42)
    u = torch.randn(num_nodes, 1)
    u = u / torch.norm(u)
    for _ in range(500):
        v = torch.sparse.mm(A_sparse, u)
        v = v / (torch.norm(v) + 1e-12)
        u = torch.sparse.mm(A_sparse.t(), v)
        u = u / (torch.norm(u) + 1e-12)
    spectral_norm = torch.norm(torch.sparse.mm(A_sparse, u)).item()
    
    assert spectral_norm <= 1.000001, f"Expected ||A_norm||_2 <= 1.0, got {spectral_norm}"
    assert spectral_norm >= 0.9999, f"Expected converged ||A_norm||_2 approx 1.0, got {spectral_norm}"


def test_cora_reachability_splits(cora_data):
    G, _, _, _ = cora_data
    splits = generate_cora_reachability_splits(G, num_train=100, num_val=30, num_test=30, seed=42)
    
    for split_name, data in splits.items():
        assert "sources" in data and "targets" in data and "labels" in data and "hops" in data
        assert len(data["sources"]) == len(data["targets"]) == len(data["labels"]) == len(data["hops"])
        # Check balance
        labels = data["labels"]
        pos_ratio = labels.float().mean().item()
        assert 0.40 <= pos_ratio <= 0.60, f"Expected balanced labels in {split_name}, got {pos_ratio}"


def test_cora_models_forward(cora_data):
    G, A_sparse, paper_ids, _ = cora_data
    num_nodes = len(paper_ids)
    
    sources = torch.tensor([0, 10, 25])
    targets = torch.tensor([5, 12, 100])
    
    # 1. Direct Baseline
    baseline = CoraDirectBaseline(num_nodes=num_nodes, embed_dim=16, hidden_dim=32)
    logits_b = baseline(sources, targets)
    assert logits_b.shape == (3, 2)
    assert torch.isfinite(logits_b).all()
    
    # 2. Discrete GNN (K=2, 4, 6, 8)
    for k in [2, 4, 6, 8]:
        gnn = CoraDiscreteGNN(num_nodes=num_nodes, node_dim=8, num_layers=k)
        logits_g = gnn(A_sparse, sources, targets)
        assert logits_g.shape == (3, 2)
        assert torch.isfinite(logits_g).all()
    
    # 3. Contractive Latent Reasoner
    clr = CoraContractiveReasoner(num_nodes=num_nodes, node_dim=8, min_damping=1.5)
    t_span = torch.tensor([0.0, 1.0])
    logits_c, z_star = clr(A_sparse, sources, targets, t_span=t_span)
    assert logits_c.shape == (3, 2)
    assert z_star.shape == (3, num_nodes, 8)
    assert torch.isfinite(logits_c).all()
    assert torch.isfinite(z_star).all()
    
    # Check Demidovich contraction bound
    d_min = torch.min(clr.get_damping()).item()
    assert d_min >= 1.5, f"Expected d_min >= 1.5, got {d_min}"


def test_cora_empirical_contraction(cora_data):
    """Numerically verifies empirical lambda_max(Sym(J)) < 0 across 43,328 dims (C-6)."""
    _, A_sparse, paper_ids, _ = cora_data
    num_nodes = len(paper_ids)
    
    model = CoraContractiveReasoner(num_nodes=num_nodes, node_dim=16, min_damping=1.5)
    empirical_lambda = compute_empirical_cora_sym_j(model, A_sparse, num_iters=10)
    
    # Empirical max eigenvalue must be strictly negative
    assert empirical_lambda < -0.5, f"Expected lambda_max(Sym(J)) < -0.5, got {empirical_lambda}"


def test_gnn_structural_horizon_truncation(cora_data):
    """
    Verifies that for hop distance h > K, the K-layer discrete GNN message-passing
    state at target node v is identically zero (null state), proving the structural truncation (C-2).
    """
    G, A_sparse, paper_ids, _ = cora_data
    num_nodes = len(paper_ids)
    
    # Find a pair with distance exactly 3
    import networkx as nx
    pair_found = None
    for u in range(100):
        lengths = nx.single_source_shortest_path_length(G, u, cutoff=4)
        for v, d in lengths.items():
            if d == 3:
                pair_found = (u, v)
                break
        if pair_found:
            break
            
    if pair_found is None:
        pytest.skip("Could not find 3-hop pair in sample.")
        
    src, tgt = pair_found
    src_t = torch.tensor([src])
    tgt_t = torch.tensor([tgt])
    
    # 2-layer GNN: cannot reach hop 3
    gnn_k2 = CoraDiscreteGNN(num_nodes=num_nodes, node_dim=8, num_layers=2)
    
    # Manually propagate through K=2 layers
    h = torch.zeros(1, num_nodes, 8)
    h[0, src, 0] = 1.0
    for layer in gnn_k2.layers:
        h_flat = h.permute(1, 0, 2).reshape(num_nodes, 8)
        prop_flat = torch.sparse.mm(A_sparse, h_flat)
        prop = prop_flat.view(num_nodes, 1, 8).permute(1, 0, 2)
        h = torch.relu(layer(prop))
        
    target_state = h[0, tgt]
    assert torch.all(target_state == 0.0), f"Expected exact null state for hop 3 with K=2, got {target_state}"


def test_saved_checkpoints_accuracy_and_behavior(cora_data):
    """
    Directly guards the numerical claims and baseline comparisons from the Cora experiment (C-7):
    - Loads persisted checkpoints from experiments/results/checkpoints/
    - Evaluates on canonical test set (300 pairs)
    - Asserts CLR test accuracy >= 0.98
    - Asserts Discrete GNN K=2 test accuracy in [0.58, 0.63] and predicts unreachable for >= 85% of pairs
    - Asserts Discrete GNN K=4 test accuracy in [0.72, 0.77] and predicts unreachable for >= 70% of pairs
    - Asserts Discrete GNN K=6 test accuracy in [0.85, 0.90] (resolves C-1)
    - Asserts Discrete GNN K=8 test accuracy in [0.84, 0.89] (over-smoothing saturation)
    """
    checkpoints_dir = Path(__file__).resolve().parents[1] / "experiments" / "results" / "checkpoints"
    clr_path = checkpoints_dir / "cora_clr.pt"
    if not clr_path.exists():
        pytest.skip("Checkpoints not yet generated; skipping checkpoint validation.")
        
    G, A_sparse, paper_ids, _ = cora_data
    num_nodes = len(paper_ids)
    
    splits = generate_cora_reachability_splits(G, num_train=1200, num_val=300, num_test=300, seed=42)
    test_data = splits["test"]
    sources = test_data["sources"]
    targets = test_data["targets"]
    labels = test_data["labels"]
    
    # 1. Evaluate CLR
    clr = CoraContractiveReasoner(num_nodes=num_nodes, node_dim=16, min_damping=1.5)
    clr.load_state_dict(torch.load(clr_path, map_location="cpu", weights_only=True))
    clr.eval()
    with torch.no_grad():
        logits_clr, _ = clr(
            A_sparse, sources, targets,
            t_span=torch.tensor([0.0, 4.0]),
            perturbation_std=0.0,
            perturbation_time=2.0
        )
        preds_clr = torch.argmax(logits_clr, dim=-1)
        acc_clr = (preds_clr == labels).float().mean().item()
    assert acc_clr >= 0.98, f"Expected CLR test acc >= 0.98, got {acc_clr}"

    # 2. Evaluate GNN K=2
    gnn2_path = checkpoints_dir / "cora_gnn_k2.pt"
    if gnn2_path.exists():
        gnn2 = CoraDiscreteGNN(num_nodes=num_nodes, node_dim=16, num_layers=2)
        gnn2.load_state_dict(torch.load(gnn2_path, map_location="cpu", weights_only=True))
        gnn2.eval()
        with torch.no_grad():
            logits_g2 = gnn2(A_sparse, sources, targets)
            preds_g2 = torch.argmax(logits_g2, dim=-1)
            acc_g2 = (preds_g2 == labels).float().mean().item()
            pred_unreach_ratio_g2 = (preds_g2 == 0).float().mean().item()
        assert 0.58 <= acc_g2 <= 0.63, f"Expected GNN K=2 acc in [0.58, 0.63], got {acc_g2}"
        assert pred_unreach_ratio_g2 >= 0.85, f"Expected GNN K=2 to predict unreachable >= 85%, got {pred_unreach_ratio_g2}"

    # 3. Evaluate GNN K=4
    gnn4_path = checkpoints_dir / "cora_gnn_k4.pt"
    if gnn4_path.exists():
        gnn4 = CoraDiscreteGNN(num_nodes=num_nodes, node_dim=16, num_layers=4)
        gnn4.load_state_dict(torch.load(gnn4_path, map_location="cpu", weights_only=True))
        gnn4.eval()
        with torch.no_grad():
            logits_g4 = gnn4(A_sparse, sources, targets)
            preds_g4 = torch.argmax(logits_g4, dim=-1)
            acc_g4 = (preds_g4 == labels).float().mean().item()
        assert 0.72 <= acc_g4 <= 0.77, f"Expected GNN K=4 acc in [0.72, 0.77], got {acc_g4}"

    # 4. Evaluate GNN K=6 (C-1)
    gnn6_path = checkpoints_dir / "cora_gnn_k6.pt"
    if gnn6_path.exists():
        gnn6 = CoraDiscreteGNN(num_nodes=num_nodes, node_dim=16, num_layers=6)
        gnn6.load_state_dict(torch.load(gnn6_path, map_location="cpu", weights_only=True))
        gnn6.eval()
        with torch.no_grad():
            logits_g6 = gnn6(A_sparse, sources, targets)
            preds_g6 = torch.argmax(logits_g6, dim=-1)
            acc_g6 = (preds_g6 == labels).float().mean().item()
        assert 0.85 <= acc_g6 <= 0.90, f"Expected GNN K=6 acc in [0.85, 0.90], got {acc_g6}"

    # 5. Evaluate GNN K=8 (Over-smoothing)
    gnn8_path = checkpoints_dir / "cora_gnn_k8.pt"
    if gnn8_path.exists():
        gnn8 = CoraDiscreteGNN(num_nodes=num_nodes, node_dim=16, num_layers=8)
        gnn8.load_state_dict(torch.load(gnn8_path, map_location="cpu", weights_only=True))
        gnn8.eval()
        with torch.no_grad():
            logits_g8 = gnn8(A_sparse, sources, targets)
            preds_g8 = torch.argmax(logits_g8, dim=-1)
            acc_g8 = (preds_g8 == labels).float().mean().item()
        assert 0.84 <= acc_g8 <= 0.89, f"Expected GNN K=8 acc in [0.84, 0.89], got {acc_g8}"


def test_cora_perturbation_monotonicity(cora_data):
    """
    Guards perturbation response via strict monotonicity (C-3):
    - Uses unified numerical integration across clean and perturbed trajectories
    - Evaluates on the canonical test set (300 pairs)
    - Asserts clean test accuracy >= 0.98
    - Asserts monotonic degradation under noise: acc(0) >= acc(1e-6) >= acc(1e-2)
    - Asserts bounded chance limit: acc(1e-2) >= 0.45 (no NaN or numerical divergence)
    """
    checkpoints_dir = Path(__file__).resolve().parents[1] / "experiments" / "results" / "checkpoints"
    clr_path = checkpoints_dir / "cora_clr.pt"
    if not clr_path.exists():
        pytest.skip("Checkpoints not yet generated; skipping perturbation test.")
        
    G, A_sparse, paper_ids, _ = cora_data
    num_nodes = len(paper_ids)
    
    splits = generate_cora_reachability_splits(G, num_train=1200, num_val=300, num_test=300, seed=42)
    test_data = splits["test"]
    sources = test_data["sources"]
    targets = test_data["targets"]
    labels = test_data["labels"]
    
    clr = CoraContractiveReasoner(num_nodes=num_nodes, node_dim=16, min_damping=1.5)
    clr.load_state_dict(torch.load(clr_path, map_location="cpu", weights_only=True))
    clr.eval()
    
    batch_size = 64
    def eval_pert(sigma: float):
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(sources), batch_size):
                s_b = sources[i : i + batch_size]
                t_b = targets[i : i + batch_size]
                logits, _ = clr(A_sparse, s_b, t_b, t_span=torch.tensor([0.0, 4.0]), perturbation_std=sigma, perturbation_time=2.0)
                all_preds.append(torch.argmax(logits, dim=-1))
        all_preds = torch.cat(all_preds, dim=0)
        return (all_preds == labels).float().mean().item()
        
    acc_clean = eval_pert(0.0)
    acc_micro = eval_pert(1e-6)
    acc_macro = eval_pert(1e-2)
    
    assert acc_clean >= 0.98, f"Expected clean accuracy >= 0.98, got {acc_clean}"
    assert acc_clean >= acc_micro >= acc_macro, (
        f"Expected monotonic degradation under noise: acc_clean ({acc_clean}) >= "
        f"acc_micro ({acc_micro}) >= acc_macro ({acc_macro})"
    )
    assert acc_macro >= 0.45, f"Expected bounded chance limit (>= 0.45), got {acc_macro}"




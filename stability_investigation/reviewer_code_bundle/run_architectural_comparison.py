"""
Phase 4 & 5: Controlled Architectural Comparison and Ablation.
Evaluates:
1. Baseline Contractive ODE (d_min = 1.5, alpha >= 0.5)
2. Candidate A (Margin-Tuned ODE: d_min = 1.05, alpha >= 0.05)
3. Candidate B (Multiscale Skip-Hop ODE: gamma1=0.6, gamma2=0.4, d_min=1.5, alpha >= 0.5)
4. Candidate C (Structural Readout Augmentation: d_min=1.5, alpha >= 0.5)
5. Baseline Discrete GNN (6 layers)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn as nn

from models import ContinuousGraphODE, DiscreteGNN, MultiscaleContractiveODE, StructuralReadoutODE
from run_damping_intervention import load_graph_instances

def compute_node_degrees(N: int, edge_index: torch.Tensor) -> torch.Tensor:
    # returns (N, 2): col 0 is out_degree, col 1 is in_degree
    degs = torch.zeros((N, 2), dtype=torch.float32)
    u_idx = edge_index[0]
    v_idx = edge_index[1]
    for u in u_idx:
        degs[u, 0] += 1.0
    for v in v_idx:
        degs[v, 1] += 1.0
    return degs

def evaluate_extended_model(model: nn.Module, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
    model.eval()
    all_preds = []
    all_targets = []
    all_hops = []
    
    with torch.no_grad():
        for inst in instances:
            P = inst["operator_P"]
            queries = inst["queries"]
            edge_index = inst["edge_index"]
            N = inst["num_nodes"]
            degs = compute_node_degrees(N, edge_index)
            
            src = torch.tensor([q["source"] for q in queries], dtype=torch.long)
            tgt = torch.tensor([q["target"] for q in queries], dtype=torch.long)
            lbl = torch.tensor([q["label"] for q in queries], dtype=torch.long)
            hps = torch.tensor([q["hop"] for q in queries], dtype=torch.long)
            
            B = len(queries)
            logits_list = []
            for i in range(0, B, 128):
                b_src = src[i:i+128]
                b_tgt = tgt[i:i+128]
                if isinstance(model, StructuralReadoutODE):
                    out = model(P, b_src, b_tgt, degrees=degs)
                else:
                    out = model(P, b_src, b_tgt)
                logits_list.append(out)
            logits = torch.cat(logits_list, dim=0)
            preds = logits.argmax(dim=-1)
            
            all_preds.append(preds)
            all_targets.append(lbl)
            all_hops.append(hps)
            
    preds_cat = torch.cat(all_preds, dim=0).numpy()
    targets_cat = torch.cat(all_targets, dim=0).numpy()
    hops_cat = torch.cat(all_hops, dim=0).numpy()
    
    total = len(targets_cat)
    correct = (preds_cat == targets_cat).sum()
    acc = float(correct) / total if total > 0 else 0.0
    
    neg_mask = (targets_cat == 0)
    neg_correct = (preds_cat[neg_mask] == 0).sum() if neg_mask.sum() > 0 else 0
    neg_total = neg_mask.sum()
    spec = float(neg_correct) / float(neg_total) if neg_total > 0 else 0.0
    
    pos_mask = (targets_cat == 1)
    pos_correct = (preds_cat[pos_mask] == 1).sum() if pos_mask.sum() > 0 else 0
    pos_total = pos_mask.sum()
    rec = float(pos_correct) / float(pos_total) if pos_total > 0 else 0.0
    
    hop_recalls = {}
    for h in range(1, 7):
        h_mask = (targets_cat == 1) & (hops_cat == h)
        h_tot = h_mask.sum()
        if h_tot > 0:
            h_cor = (preds_cat[h_mask] == 1).sum()
            hop_recalls[h] = {
                "correct": int(h_cor),
                "total": int(h_tot),
                "recall": float(h_cor) / float(h_tot)
            }
        else:
            hop_recalls[h] = {"correct": 0, "total": 0, "recall": 0.0}
            
    return {
        "accuracy": acc,
        "specificity": spec,
        "overall_recall": rec,
        "hop_recalls": hop_recalls,
        "total_queries": int(total),
        "total_pos": int(pos_total),
        "total_neg": int(neg_total)
    }

def train_and_eval_architecture(
    arch_name: str,
    train_inst: List[Dict[str, Any]],
    val_inst: List[Dict[str, Any]],
    test_inst: List[Dict[str, Any]],
    seed: int,
    epochs: int = 25,
    lr: float = 0.005,
    steps: int = 16,
    horizon: float = 4.0,
    device: str = "cpu"
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if arch_name == "Baseline_Damped_1.5":
        model = ContinuousGraphODE(
            node_dim=8, damping=True, damping_offset=1.5, method="rk4", steps=steps, horizon=horizon
        ).to(device)
    elif arch_name == "Candidate_A_Tuned_1.05":
        model = ContinuousGraphODE(
            node_dim=8, damping=True, damping_offset=1.05, method="rk4", steps=steps, horizon=horizon
        ).to(device)
    elif arch_name == "Candidate_B_Multiscale_1.5":
        model = MultiscaleContractiveODE(
            node_dim=8, gamma1=0.6, gamma2=0.4, damping_offset=1.5, method="rk4", steps=steps, horizon=horizon
        ).to(device)
    elif arch_name == "Candidate_C_Structural_1.5":
        model = StructuralReadoutODE(
            node_dim=8, damping_offset=1.5, method="rk4", steps=steps, horizon=horizon
        ).to(device)
    elif arch_name == "Baseline_Discrete_GNN6":
        model = DiscreteGNN(node_dim=8, num_layers=6).to(device)
    else:
        raise ValueError(f"Unknown arch_name: {arch_name}")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float("inf")
    best_state = None
    
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for inst in train_inst:
            P = inst["operator_P"].to(device)
            queries = inst["queries"]
            N = inst["num_nodes"]
            edge_index = inst["edge_index"]
            degs = compute_node_degrees(N, edge_index).to(device)
            
            src = torch.tensor([q["source"] for q in queries], dtype=torch.long, device=device)
            tgt = torch.tensor([q["target"] for q in queries], dtype=torch.long, device=device)
            lbl = torch.tensor([q["label"] for q in queries], dtype=torch.long, device=device)
            
            perm = torch.randperm(len(queries))
            for i in range(0, len(queries), 64):
                idx = perm[i:i+64]
                optimizer.zero_grad()
                if isinstance(model, StructuralReadoutODE):
                    logits = model(P, src[idx], tgt[idx], degrees=degs)
                else:
                    logits = model(P, src[idx], tgt[idx])
                loss = criterion(logits, lbl[idx])
                loss.backward()
                optimizer.step()
                
        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for inst in val_inst:
                P = inst["operator_P"].to(device)
                queries = inst["queries"]
                N = inst["num_nodes"]
                edge_index = inst["edge_index"]
                degs = compute_node_degrees(N, edge_index).to(device)
                
                src = torch.tensor([q["source"] for q in queries], dtype=torch.long, device=device)
                tgt = torch.tensor([q["target"] for q in queries], dtype=torch.long, device=device)
                lbl = torch.tensor([q["label"] for q in queries], dtype=torch.long, device=device)
                
                if isinstance(model, StructuralReadoutODE):
                    logits = model(P, src, tgt, degrees=degs)
                else:
                    logits = model(P, src, tgt)
                loss = criterion(logits, lbl)
                val_loss += loss.item() * len(queries)
                val_count += len(queries)
        mean_val = val_loss / val_count if val_count > 0 else float("inf")
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    train_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
        
    # Evaluate on held-out test instances
    test_metrics = evaluate_extended_model(model, test_inst)
    
    # Extract realized stability parameters
    sigma_b1, sigma_b2, d_min_realized, alpha_realized = 0.0, 0.0, 0.0, 0.0
    if isinstance(model, (ContinuousGraphODE, StructuralReadoutODE)):
        with torch.no_grad():
            w1, w2, d = model.field_parameters()
            sigma_b1 = float(torch.linalg.svdvals(w1)[0].item())
            sigma_b2 = float(torch.linalg.svdvals(w2)[0].item())
            d_min_realized = float(d.min().item())
            alpha_realized = d_min_realized - (sigma_b1 * sigma_b2)
    elif isinstance(model, MultiscaleContractiveODE):
        with torch.no_grad():
            w11, w12, w21, w22, d = model.field_parameters()
            s11 = float(torch.linalg.svdvals(w11)[0].item())
            s12 = float(torch.linalg.svdvals(w12)[0].item())
            s21 = float(torch.linalg.svdvals(w21)[0].item())
            s22 = float(torch.linalg.svdvals(w22)[0].item())
            d_min_realized = float(d.min().item())
            alpha_realized = d_min_realized - (model.gamma1 * s11 * s12 + model.gamma2 * s21 * s22)
            sigma_b1 = (s11 + s21) / 2.0
            sigma_b2 = (s12 + s22) / 2.0
            
    return {
        "architecture": arch_name,
        "seed": seed,
        "train_time_sec": train_time,
        "best_val_loss": best_val_loss,
        "sigma_b1": sigma_b1,
        "sigma_b2": sigma_b2,
        "d_min_realized": d_min_realized,
        "alpha_realized": alpha_realized,
        "test_metrics": test_metrics,
        "checkpoint_state": best_state
    }

def run_architectural_comparison():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / "synthetic"
    results_dir = base_dir / "results" / "architectural_comparison"
    ckpt_dir = results_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    families = ["chain_ladder", "bottleneck", "random_dag"]
    seeds = [42, 101, 777]
    
    architectures = [
        "Baseline_Damped_1.5",
        "Candidate_A_Tuned_1.05",
        "Candidate_B_Multiscale_1.5",
        "Candidate_C_Structural_1.5",
        "Baseline_Discrete_GNN6"
    ]
    
    all_results = {
        "study": "Phase 4 & 5 Controlled Architectural Comparison",
        "date": "2026-11-16",
        "families": {}
    }
    
    for fam in families:
        print(f"\n=======================================================")
        print(f"Architectural Comparison on Family: {fam}")
        print(f"=======================================================")
        fam_dir = data_dir / fam
        train_inst = load_graph_instances(fam_dir, "train")
        val_inst = load_graph_instances(fam_dir, "val")
        test_inst = load_graph_instances(fam_dir, "test")
        
        all_results["families"][fam] = {}
        
        for arch in architectures:
            print(f"--> Architecture: {arch}")
            runs = []
            for s in seeds:
                res = train_and_eval_architecture(
                    arch_name=arch,
                    train_inst=train_inst,
                    val_inst=val_inst,
                    test_inst=test_inst,
                    seed=s
                )
                ckpt_path = ckpt_dir / f"{fam}_{arch}_seed_{s}.pt"
                torch.save(res["checkpoint_state"], ckpt_path)
                del res["checkpoint_state"]
                runs.append(res)
            all_results["families"][fam][arch] = runs
            
    out_json = results_dir / "architectural_comparison_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nArchitectural comparison complete. Results written to {out_json}")

if __name__ == "__main__":
    run_architectural_comparison()

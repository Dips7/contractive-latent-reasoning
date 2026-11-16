"""
Phase 3: Controlled Damping Sweep Experiment.
Exploratory sweep evaluating associations between nominal damping parameters (contraction margin lower bound),
loss progression, and multi-hop reachability recall across synthetic graph families under fixed training budgets.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure repository root is on path

from models import ContinuousGraphODE, DiscreteGNN

def load_graph_instances(fam_dir: Path, split_name: str) -> List[Dict[str, Any]]:
    files = sorted(fam_dir.glob(f"{split_name}_inst_*.pt"))
    instances = []
    for f in files:
        rec = torch.load(f, weights_only=False)
        instances.append(rec)
    return instances

def evaluate_on_instances(model: nn.Module, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
    model.eval()
    all_preds = []
    all_targets = []
    all_hops = []
    
    with torch.no_grad():
        for inst in instances:
            P = inst["operator_P"]
            queries = inst["queries"]
            src = torch.tensor([q["source"] for q in queries], dtype=torch.long)
            tgt = torch.tensor([q["target"] for q in queries], dtype=torch.long)
            lbl = torch.tensor([q["label"] for q in queries], dtype=torch.long)
            hps = torch.tensor([q["hop"] for q in queries], dtype=torch.long)
            
            # Forward in batches of 128
            B = len(queries)
            logits_list = []
            for i in range(0, B, 128):
                b_src = src[i:i+128]
                b_tgt = tgt[i:i+128]
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
    
    # Negative specificity (hop 0, label 0)
    neg_mask = (targets_cat == 0)
    neg_correct = (preds_cat[neg_mask] == 0).sum() if neg_mask.sum() > 0 else 0
    neg_total = neg_mask.sum()
    spec = float(neg_correct) / float(neg_total) if neg_total > 0 else 0.0
    
    # Positive recall (label 1)
    pos_mask = (targets_cat == 1)
    pos_correct = (preds_cat[pos_mask] == 1).sum() if pos_mask.sum() > 0 else 0
    pos_total = pos_mask.sum()
    rec = float(pos_correct) / float(pos_total) if pos_total > 0 else 0.0
    
    # Hop-by-hop recall
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

def train_and_evaluate_model(
    model_type: str,
    damping_offset: float,
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
    # Set seed strictly BEFORE model creation
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if model_type == "ContinuousDamped":
        model = ContinuousGraphODE(
            node_dim=8,
            damping=True,
            damping_offset=damping_offset,
            method="rk4",
            steps=steps,
            horizon=horizon,
            readout_type="log_norm"
        ).to(device)
    elif model_type == "ContinuousUndamped":
        model = ContinuousGraphODE(
            node_dim=8,
            damping=False,
            method="rk4",
            steps=steps,
            horizon=horizon,
            readout_type="log_norm"
        ).to(device)
    elif model_type == "DiscreteGNN":
        model = DiscreteGNN(
            node_dim=8,
            num_layers=6,
            readout_type="log_norm"
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float("inf")
    best_state = None
    
    # Train loop
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for inst in train_inst:
            P = inst["operator_P"].to(device)
            queries = inst["queries"]
            src = torch.tensor([q["source"] for q in queries], dtype=torch.long, device=device)
            tgt = torch.tensor([q["target"] for q in queries], dtype=torch.long, device=device)
            lbl = torch.tensor([q["label"] for q in queries], dtype=torch.long, device=device)
            
            # Mini-batch shuffle
            perm = torch.randperm(len(queries))
            for i in range(0, len(queries), 64):
                idx = perm[i:i+64]
                optimizer.zero_grad()
                logits = model(P, src[idx], tgt[idx])
                loss = criterion(logits, lbl[idx])
                loss.backward()
                optimizer.step()
                
        # Validation loss
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for inst in val_inst:
                P = inst["operator_P"].to(device)
                queries = inst["queries"]
                src = torch.tensor([q["source"] for q in queries], dtype=torch.long, device=device)
                tgt = torch.tensor([q["target"] for q in queries], dtype=torch.long, device=device)
                lbl = torch.tensor([q["label"] for q in queries], dtype=torch.long, device=device)
                logits = model(P, src, tgt)
                loss = criterion(logits, lbl)
                val_loss += loss.item() * len(queries)
                val_count += len(queries)
        mean_val = val_loss / val_count if val_count > 0 else float("inf")
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    train_time = time.time() - t0
    
    # Load best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        
    # Measure realized stability parameters
    sigma_b1 = 0.0
    sigma_b2 = 0.0
    d_min_realized = 0.0
    alpha_realized = 0.0
    
    if isinstance(model, ContinuousGraphODE):
        with torch.no_grad():
            w1, w2, d = model.field_parameters()
            sigma_b1 = float(torch.linalg.svdvals(w1)[0].item())
            sigma_b2 = float(torch.linalg.svdvals(w2)[0].item())
            if model.damped:
                d_min_realized = float(d.min().item())
                alpha_realized = d_min_realized - (sigma_b1 * sigma_b2)
            else:
                d_min_realized = 0.0
                alpha_realized = 0.0 - (sigma_b1 * sigma_b2)
                
    # Evaluate on held-out test instances
    test_metrics = evaluate_on_instances(model, test_inst)
    
    return {
        "model_type": model_type,
        "nominal_damping": damping_offset if model_type == "ContinuousDamped" else 0.0,
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

def run_all_experiments():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / "synthetic"
    results_dir = base_dir / "results" / "damping_intervention"
    ckpt_dir = results_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    families = ["chain_ladder", "bottleneck", "random_dag"]
    seeds = [42, 101, 777]
    
    # Damping sweep grid: from negative margin (-0.5) to strong contraction (+1.0)
    damping_offsets = [0.5, 0.8, 1.05, 1.2, 1.5, 2.0]
    
    all_results: Dict[str, Any] = {
        "benchmark": "Phase 3 Controlled Damping Intervention",
        "date": "2026-11-16",
        "families": {}
    }
    
    for fam in families:
        print(f"\n=======================================================")
        print(f"Running Damping Intervention on Family: {fam}")
        print(f"=======================================================")
        fam_dir = data_dir / fam
        train_inst = load_graph_instances(fam_dir, "train")
        val_inst = load_graph_instances(fam_dir, "val")
        test_inst = load_graph_instances(fam_dir, "test")
        
        all_results["families"][fam] = {}
        
        # 1. Sweep Damped Continuous ODE across damping offsets
        for d_off in damping_offsets:
            cfg_key = f"ContinuousDamped_dmin_{d_off:.2f}"
            print(f"--> Config: {cfg_key}")
            runs = []
            for s in seeds:
                res = train_and_evaluate_model(
                    model_type="ContinuousDamped",
                    damping_offset=d_off,
                    train_inst=train_inst,
                    val_inst=val_inst,
                    test_inst=test_inst,
                    seed=s
                )
                ckpt_path = ckpt_dir / f"{fam}_{cfg_key}_seed_{s}.pt"
                torch.save(res["checkpoint_state"], ckpt_path)
                del res["checkpoint_state"] # Don't keep weights in JSON
                runs.append(res)
            all_results["families"][fam][cfg_key] = runs
            
        # 2. Undamped Continuous ODE
        cfg_key = "ContinuousUndamped"
        print(f"--> Config: {cfg_key}")
        runs = []
        for s in seeds:
            res = train_and_evaluate_model(
                model_type="ContinuousUndamped",
                damping_offset=0.0,
                train_inst=train_inst,
                val_inst=val_inst,
                test_inst=test_inst,
                seed=s
            )
            ckpt_path = ckpt_dir / f"{fam}_{cfg_key}_seed_{s}.pt"
            torch.save(res["checkpoint_state"], ckpt_path)
            del res["checkpoint_state"]
            runs.append(res)
        all_results["families"][fam][cfg_key] = runs
        
        # 3. Discrete GNN (6 layers)
        cfg_key = "DiscreteGNN_L6"
        print(f"--> Config: {cfg_key}")
        runs = []
        for s in seeds:
            res = train_and_evaluate_model(
                model_type="DiscreteGNN",
                damping_offset=0.0,
                train_inst=train_inst,
                val_inst=val_inst,
                test_inst=test_inst,
                seed=s
            )
            ckpt_path = ckpt_dir / f"{fam}_{cfg_key}_seed_{s}.pt"
            torch.save(res["checkpoint_state"], ckpt_path)
            del res["checkpoint_state"]
            runs.append(res)
        all_results["families"][fam][cfg_key] = runs
        
    # Save full JSON
    out_json = results_dir / "damping_intervention_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll experiments complete. Results written to {out_json}")

if __name__ == "__main__":
    run_all_experiments()

"""
Reproducible Test Loss Evaluation and Checkpoint Digest Persistence Script.
Computes:
1. Exact held-out cross-entropy test loss for all 45 architectural comparison checkpoints.
2. Exact held-out cross-entropy test loss for all 72 damping intervention checkpoints.
3. SHA-256 digests for all 117 evaluated checkpoint files and test dataset files.
4. Updates JSON result records and generates audit certificates with full provenance.
"""

import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn as nn

from models import ContinuousGraphODE, DiscreteGNN, MultiscaleContractiveODE, StructuralReadoutODE
from run_damping_intervention import load_graph_instances

def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def compute_node_degrees(N: int, edge_index: torch.Tensor) -> torch.Tensor:
    degs = torch.zeros((N, 2), dtype=torch.float32)
    for u in edge_index[0]: degs[u, 0] += 1.0
    for v in edge_index[1]: degs[v, 1] += 1.0
    return degs

def evaluate_model_loss(model: nn.Module, test_instances: List[Dict[str, Any]]) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_queries = 0
    with torch.no_grad():
        for inst in test_instances:
            P = inst["operator_P"]
            queries = inst["queries"]
            src = torch.tensor([q["source"] for q in queries], dtype=torch.long)
            tgt = torch.tensor([q["target"] for q in queries], dtype=torch.long)
            lbl = torch.tensor([q["label"] for q in queries], dtype=torch.long)
            
            if isinstance(model, StructuralReadoutODE):
                edge_index = inst["edge_index"]
                N = inst["num_nodes"]
                degs = compute_node_degrees(N, edge_index)
                logits = model(P, src, tgt, degrees=degs)
            else:
                logits = model(P, src, tgt)
                
            loss = criterion(logits, lbl)
            total_loss += float(loss.item()) * len(queries)
            total_queries += len(queries)
            
    return total_loss / total_queries if total_queries > 0 else 0.0

def run_evaluation():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / "synthetic"
    
    # 1. Dataset provenance
    manifest_path = data_dir / "benchmark_manifest.json"
    if not manifest_path.exists():
        manifest_path = base_dir / "benchmark_manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    
    test_files_sha256 = {}
    for fam in ["chain_ladder", "bottleneck", "random_dag"]:
        fam_test_files = sorted((data_dir / fam).glob("test_inst_*.pt"))
        for tf in fam_test_files:
            test_files_sha256[f"{fam}/{tf.name}"] = sha256_file(tf)
            
    print(f"Dataset manifest SHA-256: {manifest_sha256}")
    print(f"Hashed {len(test_files_sha256)} held-out test data files.")

    # 2. Architectural Comparison Checkpoints
    arch_results_path = base_dir / "results" / "architectural_comparison" / "architectural_comparison_results.json"
    arch_ckpt_dir = base_dir / "results" / "architectural_comparison" / "checkpoints"
    
    with open(arch_results_path, "r") as f:
        arch_data = json.load(f)
        
    arch_certificate = {
        "evaluation_title": "Phase 4 & 5 Architectural Comparison Held-Out Test Loss Audit",
        "timestamp": "2026-11-16",
        "command": "python3 stability_investigation/scripts/evaluate_and_persist_test_loss.py",
        "dataset_manifest_sha256": manifest_sha256,
        "test_dataset_files": test_files_sha256,
        "checkpoints": {},
        "summary_table": {}
    }
    
    architectures = [
        "Baseline_Damped_1.5",
        "Candidate_A_Tuned_1.05",
        "Candidate_B_Multiscale_1.5",
        "Candidate_C_Structural_1.5",
        "Baseline_Discrete_GNN6"
    ]
    
    for fam in ["chain_ladder", "bottleneck", "random_dag"]:
        test_inst = load_graph_instances(data_dir / fam, "test")
        arch_certificate["summary_table"][fam] = {}
        
        for arch in architectures:
            runs = arch_data["families"][fam][arch]
            loss_list = []
            
            for r in runs:
                s = r["seed"]
                ckpt_name = f"{fam}_{arch}_seed_{s}.pt"
                ckpt_p = arch_ckpt_dir / ckpt_name
                assert ckpt_p.exists(), f"Missing checkpoint {ckpt_p}"
                
                ckpt_hash = sha256_file(ckpt_p)
                state = torch.load(ckpt_p, weights_only=False)
                
                if arch == "Baseline_Damped_1.5":
                    m = ContinuousGraphODE(node_dim=8, damping=True, damping_offset=1.5, steps=16)
                elif arch == "Candidate_A_Tuned_1.05":
                    m = ContinuousGraphODE(node_dim=8, damping=True, damping_offset=1.05, steps=16)
                elif arch == "Candidate_B_Multiscale_1.5":
                    m = MultiscaleContractiveODE(node_dim=8, gamma1=0.6, gamma2=0.4, damping_offset=1.5, steps=16)
                elif arch == "Candidate_C_Structural_1.5":
                    m = StructuralReadoutODE(node_dim=8, damping_offset=1.5, steps=16)
                elif arch == "Baseline_Discrete_GNN6":
                    m = DiscreteGNN(node_dim=8, num_layers=6)
                    
                m.load_state_dict(state)
                t_loss = evaluate_model_loss(m, test_inst)
                
                # Persist directly in run dictionary
                r["test_loss"] = float(t_loss)
                r["test_metrics"]["test_loss"] = float(t_loss)
                r["checkpoint_sha256"] = ckpt_hash
                loss_list.append(t_loss)
                
                arch_certificate["checkpoints"][ckpt_name] = {
                    "family": fam,
                    "architecture": arch,
                    "seed": s,
                    "sha256": ckpt_hash,
                    "test_loss": float(t_loss),
                    "best_val_loss": r["best_val_loss"]
                }
                
            arch_certificate["summary_table"][fam][arch] = {
                "mean_test_loss": float(np.mean(loss_list)),
                "std_test_loss": float(np.std(loss_list, ddof=1)),
                "mean_val_loss": float(np.mean([r["best_val_loss"] for r in runs])),
                "std_val_loss": float(np.std([r["best_val_loss"] for r in runs], ddof=1)),
                "per_seed_test_loss": [float(x) for x in loss_list]
            }
            
    # Save updated architectural_comparison_results.json
    with open(arch_results_path, "w") as f:
        json.dump(arch_data, f, indent=2)
    print(f"Updated {arch_results_path} with per-run test losses.")
    
    # Save architectural test loss certificate
    arch_cert_path = base_dir / "results" / "architectural_comparison" / "test_loss_evaluation_certificate.json"
    with open(arch_cert_path, "w") as f:
        json.dump(arch_certificate, f, indent=2)
    print(f"Saved architectural certificate to {arch_cert_path}")

    # 3. Damping Intervention Checkpoints
    damping_results_path = base_dir / "results" / "damping_intervention" / "damping_intervention_results.json"
    damping_ckpt_dir = base_dir / "results" / "damping_intervention" / "checkpoints"
    
    with open(damping_results_path, "r") as f:
        damping_data = json.load(f)
        
    damping_certificate = {
        "evaluation_title": "Phase 3 Damping Intervention Held-Out Test Loss Audit",
        "timestamp": "2026-11-16",
        "command": "python3 stability_investigation/scripts/evaluate_and_persist_test_loss.py",
        "dataset_manifest_sha256": manifest_sha256,
        "test_dataset_files": test_files_sha256,
        "checkpoints": {},
        "summary_table": {}
    }
    
    damping_cfgs = [
        ("ContinuousDamped_dmin_0.50", "ContinuousDamped", 0.50),
        ("ContinuousDamped_dmin_0.80", "ContinuousDamped", 0.80),
        ("ContinuousDamped_dmin_1.05", "ContinuousDamped", 1.05),
        ("ContinuousDamped_dmin_1.20", "ContinuousDamped", 1.20),
        ("ContinuousDamped_dmin_1.50", "ContinuousDamped", 1.50),
        ("ContinuousDamped_dmin_2.00", "ContinuousDamped", 2.00),
        ("ContinuousUndamped", "ContinuousUndamped", 0.0),
        ("DiscreteGNN_L6", "DiscreteGNN", 0.0)
    ]
    
    for fam in ["chain_ladder", "bottleneck", "random_dag"]:
        test_inst = load_graph_instances(data_dir / fam, "test")
        damping_certificate["summary_table"][fam] = {}
        
        for cfg_key, m_type, d_off in damping_cfgs:
            runs = damping_data["families"][fam][cfg_key]
            loss_list = []
            
            for r in runs:
                s = r["seed"]
                ckpt_name = f"{fam}_{cfg_key}_seed_{s}.pt"
                ckpt_p = damping_ckpt_dir / ckpt_name
                assert ckpt_p.exists(), f"Missing checkpoint {ckpt_p}"
                
                ckpt_hash = sha256_file(ckpt_p)
                state = torch.load(ckpt_p, weights_only=False)
                
                if m_type == "ContinuousDamped":
                    m = ContinuousGraphODE(node_dim=8, damping=True, damping_offset=d_off, steps=16)
                elif m_type == "ContinuousUndamped":
                    m = ContinuousGraphODE(node_dim=8, damping=False, steps=16)
                elif m_type == "DiscreteGNN":
                    m = DiscreteGNN(node_dim=8, num_layers=6)
                    
                m.load_state_dict(state)
                t_loss = evaluate_model_loss(m, test_inst)
                
                r["test_loss"] = float(t_loss)
                r["test_metrics"]["test_loss"] = float(t_loss)
                r["checkpoint_sha256"] = ckpt_hash
                loss_list.append(t_loss)
                
                damping_certificate["checkpoints"][ckpt_name] = {
                    "family": fam,
                    "config": cfg_key,
                    "seed": s,
                    "sha256": ckpt_hash,
                    "test_loss": float(t_loss),
                    "best_val_loss": r["best_val_loss"]
                }
                
            damping_certificate["summary_table"][fam][cfg_key] = {
                "mean_test_loss": float(np.mean(loss_list)),
                "std_test_loss": float(np.std(loss_list, ddof=1)),
                "mean_val_loss": float(np.mean([r["best_val_loss"] for r in runs])),
                "std_val_loss": float(np.std([r["best_val_loss"] for r in runs], ddof=1)),
                "per_seed_test_loss": [float(x) for x in loss_list]
            }
            
    with open(damping_results_path, "w") as f:
        json.dump(damping_data, f, indent=2)
    print(f"Updated {damping_results_path} with per-run test losses.")
    
    damping_cert_path = base_dir / "results" / "damping_intervention" / "damping_test_loss_certificate.json"
    with open(damping_cert_path, "w") as f:
        json.dump(damping_certificate, f, indent=2)
    print(f"Saved damping certificate to {damping_cert_path}")
    print("\nAll 117 checkpoints successfully audited, evaluated, and persisted.")

if __name__ == "__main__":
    run_evaluation()

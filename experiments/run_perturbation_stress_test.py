"""
Move 1: The Perturbation / Anti-Hallucination Stress Test (Calibrated Relative Noise).

Scientific Objective:
Demonstrate that when intermediate states are perturbed by noise relative to signal scale
(e.g., 20%, 50%, 100%, 200% of state standard deviation):
1. In uncontracted recurrent models, the perturbation persists and degrades accuracy.
2. In the Contractive Latent Reasoner (CLR), the Demidovich condition exponentially shrinks
   the perturbation distance to near zero:
       ||z_noisy(t) - z_clean(t)|| -> 0
   fully restoring accuracy to clean baseline levels.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time

from experiments.run_graph_reachability_experiment import (
    generate_balanced_graph_dataset,
    unpack_graph_features,
)
from src.dynamics.solvers import solve_ode_rk4
from src.utils import set_seed, save_experiment_results


# ----------------------------------------------------------------------
# 1. Uncontracted Discrete Recurrent Baseline (8 Steps)
# ----------------------------------------------------------------------
class UncontractedRecurrentBaseline(nn.Module):
    def __init__(self, num_nodes: int, node_dim: int = 16, num_steps: int = 8):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.num_steps = num_steps
        
        self.w1 = nn.Linear(node_dim, node_dim, bias=False)
        self.w2 = nn.Linear(node_dim, node_dim, bias=False)
        
        self.readout = nn.Sequential(
            nn.Linear(node_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )

    def forward(self, x, noise_step: int = -1, rel_noise: float = 0.0):
        batch_size = x.shape[0]
        device = x.device
        A, u_onehot, v_onehot = unpack_graph_features(x, self.num_nodes)
        
        deg = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
        A_norm = A / deg
        
        z = torch.zeros(batch_size, self.num_nodes, self.node_dim, device=device)
        z[:, :, 0] = u_onehot
        
        for step in range(self.num_steps):
            z_prop = torch.bmm(A_norm.transpose(1, 2), z)
            delta = self.w2(torch.tanh(self.w1(z_prop)))
            z = z + delta
            
            # Midpoint noise injection at step = num_steps // 2
            if step == noise_step and rel_noise > 0.0:
                sigma = z.std() * rel_noise
                noise = torch.randn_like(z) * sigma
                z = z + noise
                
        v_state = (z * v_onehot.unsqueeze(-1)).sum(dim=1)
        return self.readout(v_state), z


# ----------------------------------------------------------------------
# 2. Continuous Contractive Latent Reasoner (CLR)
# ----------------------------------------------------------------------
class ContractiveReasonerWithPerturbation(nn.Module):
    def __init__(self, num_nodes: int, node_dim: int = 16, min_damping: float = 1.5):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.min_damping = min_damping
        
        self.w1 = nn.utils.parametrizations.spectral_norm(nn.Linear(node_dim, node_dim, bias=False))
        self.w2 = nn.utils.parametrizations.spectral_norm(nn.Linear(node_dim, node_dim, bias=False))
        self.log_d = nn.Parameter(torch.ones(node_dim) * 0.0)
        
        self.readout = nn.Sequential(
            nn.Linear(node_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )

    def get_damping(self):
        return F.softplus(self.log_d) + self.min_damping

    def forward(self, x, t_total: float = 8.0, perturb_at: float = -1.0, rel_noise: float = 0.0):
        batch_size = x.shape[0]
        device = x.device
        A, u_onehot, v_onehot = unpack_graph_features(x, self.num_nodes)
        
        deg = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
        A_norm = A / deg
        
        source = torch.zeros(batch_size, self.num_nodes, self.node_dim, device=device)
        source[:, :, 0] = u_onehot
        z0 = source.clone()
        d = self.get_damping()
        
        def vf(t, z):
            z_prop = torch.bmm(A_norm.transpose(1, 2), z)
            flow = self.w2(torch.tanh(self.w1(z_prop)))
            return flow - d * z + source
            
        if perturb_at > 0.0 and rel_noise > 0.0:
            # Phase 1: Integrate 0 to perturb_at
            t_span_1 = torch.tensor([0.0, perturb_at], device=device)
            steps_1 = max(15, int(perturb_at * 10))
            traj_1 = solve_ode_rk4(vf, z0, t_span_1, steps=steps_1)
            z_mid = traj_1[-1]
            
            # Calibrate noise relative to state standard deviation
            sigma = z_mid.std() * rel_noise
            noise = torch.randn_like(z_mid) * sigma
            z_mid_noisy = z_mid + noise
            
            # Phase 2: Self-Healing integration to t_total
            t_span_2 = torch.tensor([perturb_at, t_total], device=device)
            steps_2 = max(25, int((t_total - perturb_at) * 10))
            traj_2 = solve_ode_rk4(vf, z_mid_noisy, t_span_2, steps=steps_2)
            z_star = traj_2[-1]
            d_init = torch.norm(noise, dim=-1).mean().item()
        else:
            t_span = torch.tensor([0.0, t_total], device=device)
            steps = max(30, int(t_total * 10))
            traj = solve_ode_rk4(vf, z0, t_span, steps=steps)
            z_star = traj[-1]
            d_init = 0.0
            
        v_state = (z_star * v_onehot.unsqueeze(-1)).sum(dim=1)
        return self.readout(v_state), z_star, d_init


def main():
    start_time = time.time()
    print("=" * 85)
    print("MOVE 1: THE PERTURBATION & SELF-HEALING BENCHMARK (RELATIVE NOISE)")
    print("=" * 85)
    
    set_seed(42)
    device = torch.device("cpu")
    num_nodes = 16
    
    print("\n[Step 1] Generating Balanced Graph Reachability Dataset (3000 train, 600 val)...")
    x_train, y_train = generate_balanced_graph_dataset(num_samples=3000, num_nodes=num_nodes, seed=42)
    x_val, y_val = generate_balanced_graph_dataset(num_samples=600, num_nodes=num_nodes, seed=100)
    
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=32, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    
    # ------------------------------------------------------------------
    # 1. Train Uncontracted Recurrent Baseline
    # ------------------------------------------------------------------
    print("\n[Step 2] Training Uncontracted Recurrent Baseline (8 steps)...")
    base_model = UncontractedRecurrentBaseline(num_nodes=num_nodes, node_dim=16, num_steps=8).to(device)
    base_opt = torch.optim.AdamW(base_model.parameters(), lr=8e-3, weight_decay=1e-4)
    
    for ep in range(1, 9):
        base_model.train()
        for bx, by in train_loader:
            base_opt.zero_grad()
            out, _ = base_model(bx)
            loss = criterion(out, by)
            loss.backward()
            base_opt.step()
            
    base_model.eval()
    corr = 0
    with torch.no_grad():
        for vx, vy in val_loader:
            out, _ = base_model(vx)
            corr += (out.argmax(-1) == vy).sum().item()
    base_clean_acc = (corr / len(y_val)) * 100
    print(f"  -> Baseline Clean Accuracy (No Noise): {base_clean_acc:.2f}%")
    
    # ------------------------------------------------------------------
    # 2. Train Contractive Latent Reasoner (CLR)
    # ------------------------------------------------------------------
    print("\n[Step 3] Training Contractive Latent Reasoner (CLR, T=4.0)...")
    clr_model = ContractiveReasonerWithPerturbation(num_nodes=num_nodes, node_dim=16, min_damping=1.5).to(device)
    clr_opt = torch.optim.AdamW(clr_model.parameters(), lr=8e-3, weight_decay=1e-4)
    
    for ep in range(1, 9):
        clr_model.train()
        for bx, by in train_loader:
            clr_opt.zero_grad()
            out, _, _ = clr_model(bx, t_total=4.0)
            loss = criterion(out, by)
            loss.backward()
            clr_opt.step()
            
    clr_model.eval()
    corr = 0
    with torch.no_grad():
        for vx, vy in val_loader:
            out, z_clean_final, _ = clr_model(vx, t_total=8.0)
            corr += (out.argmax(-1) == vy).sum().item()
    clr_clean_acc = (corr / len(y_val)) * 100
    print(f"  -> CLR Clean Accuracy (No Noise, T=8.0): {clr_clean_acc:.2f}%")
    
    # ------------------------------------------------------------------
    # 3. Relative Perturbation Stress Test
    # ------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("RELATIVE MIDPOINT PERTURBATION SWEEP: ε ~ N(0, (rel_noise * std(z))² I)")
    print("Baseline: Perturbed at Step 4 / 8")
    print("CLR:      Perturbed at t = 3.0 / 8.0 (Self-healing window Δt = 5.0)")
    print("=" * 85)
    print(f"{'Rel Noise':<10} | {'Base Acc':<10} | {'CLR Acc':<10} | {'Initial Error':<14} | {'Residual Error (t=8)':<22} | {'Decay Factor'}")
    print("-" * 85)
    
    noise_ratios = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]
    
    torch.manual_seed(42)
    sweep_data = []
    for rel_noise in noise_ratios:
        # Evaluate Baseline
        base_corr = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                out, _ = base_model(vx, noise_step=4, rel_noise=rel_noise)
                base_corr += (out.argmax(-1) == vy).sum().item()
        base_acc = (base_corr / len(y_val)) * 100
        
        # Evaluate CLR and measure distance decay
        clr_corr = 0
        init_errors = []
        residual_errors = []
        with torch.no_grad():
            for vx, vy in val_loader:
                out_clean, z_clean, _ = clr_model(vx, t_total=8.0)
                out_noisy, z_noisy, d_init = clr_model(vx, t_total=8.0, perturb_at=3.0, rel_noise=rel_noise)
                
                clr_corr += (out_noisy.argmax(-1) == vy).sum().item()
                init_errors.append(d_init)
                res_err = torch.norm(z_noisy - z_clean, dim=-1).mean().item()
                residual_errors.append(res_err)

        clr_acc = (clr_corr / len(y_val)) * 100
        mean_init = float(np.mean(init_errors)) if len(init_errors) > 0 else 0.0
        mean_residual = float(np.mean(residual_errors)) if len(residual_errors) > 0 else 0.0
        if mean_residual > 0 and mean_init > 0:
            decay_factor = float(mean_init / mean_residual)
        else:
            decay_factor = None

        sweep_data.append({
            "rel_noise": rel_noise,
            "baseline_accuracy": base_acc,
            "clr_accuracy": clr_acc,
            "initial_error": float(mean_init),
            "residual_error": float(mean_residual),
            "decay_factor": decay_factor,
        })
        decay_str = f"{decay_factor:,.1f}x" if decay_factor is not None else "N/A"
        print(f"{rel_noise*100:6.0f}%    | {base_acc:6.2f}%   | {clr_acc:6.2f}%   | {mean_init:<14.6f} | {mean_residual:<22.8f} | {decay_str}")
        
    print("=" * 85)

    save_experiment_results(
        experiment_name="perturbation_stress_test",
        metrics={
            "sweep_results": sweep_data,
            "clr_clean_acc": clr_clean_acc,
        },
        config={
            "num_nodes": num_nodes,
            "seed": 42,
        },
        seed=42,
        start_time=start_time,
    )


if __name__ == "__main__":
    main()

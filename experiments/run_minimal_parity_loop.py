"""
Minimal End-to-End Falsification Loop:
N-bit Parity + Approach A (ICNN + Positive Damping) with Online Contraction Tracking.

Objective:
Quickly determine if Approach A can learn parity without NaNs, while verifying that
the Demidovich contraction bound holds dynamically on trained weights at every interval.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time

from data.generators.parity import generate_parity_dataset
from src.dynamics.energy import InputConvexPotential
from src.dynamics.vector_field import DampedGradientFlowField
from src.dynamics.solvers import solve_ode_rk4
from src.dynamics.contraction import verify_demidovich_condition
from src.utils import set_seed, save_experiment_results


class MinimalContractiveReasoner(nn.Module):
    """
    Smallest viable pipeline:
    Bit Sequence x -> GRU Encoder -> Context c -> ICNN + Damping Vector Field -> z* -> Linear Readout
    """
    def __init__(self, seq_len: int, latent_dim: int = 32, context_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        
        # 1. Simple BiGRU Encoder
        self.encoder = nn.GRU(
            input_size=1,
            hidden_size=context_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        
        # 2. ICNN Potential (Guaranteed Convex in z)
        self.energy = InputConvexPotential(
            latent_dim=latent_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
        )
        
        # 3. Strictly Damped Gradient Flow Field (Demidovich Contractive by construction)
        self.vector_field = DampedGradientFlowField(
            energy_model=self.energy,
            latent_dim=latent_dim,
            damping_init=0.8,
            min_damping=0.1, # Guaranteed kappa >= 0.1
        )
        
        # 4. Linear Probe Readout directly on contracted state z*
        self.probe = nn.Linear(latent_dim, 2)

    def forward(self, x: torch.Tensor, t_span: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        # x shape: (B, seq_len, 1)
        _, h_n = self.encoder(x)
        c = torch.cat([h_n[0], h_n[1]], dim=-1) # (B, context_dim)
        
        # Set context for ODE integration
        self.vector_field.set_context(c)
        z0 = torch.zeros(batch_size, self.latent_dim, device=x.device)
        
        # Integrate via RK4 solver
        traj = solve_ode_rk4(self.vector_field, z0, t_span, steps=20)
        z_star = traj[-1]
        
        # Readout logits
        logits = self.probe(z_star)
        return logits, z_star


def run_online_contraction_check(model: MinimalContractiveReasoner, sample_x: torch.Tensor, device: torch.device):
    """
    Evaluates contraction on real, currently trained weights:
    1. lambda_max(Sym(J)) <= -kappa
    2. Empirical multi-seed trajectory distance decay: ||z1(T) - z2(T)|| / ||z1(0) - z2(0)||
    """
    model.eval()
    with torch.no_grad():
        _, h_n = model.encoder(sample_x[:4])
        c_test = torch.cat([h_n[0], h_n[1]], dim=-1)
        
    z_samples = torch.randn(4, model.latent_dim, device=device)
    
    # 1. Eigenspectrum check
    results = verify_demidovich_condition(model.vector_field, z_samples, c_test, kappa_target=0.0)
    max_eig = results["max_eigenvalue"]
    
    # 2. Multi-seed trajectory divergence test
    # Sample two different initial guesses z1(0), z2(0)
    z1_0 = torch.randn(1, model.latent_dim, device=device) * 2.0
    z2_0 = torch.randn(1, model.latent_dim, device=device) * 2.0
    init_dist = torch.norm(z1_0 - z2_0).item()
    
    model.vector_field.set_context(c_test[0:1])
    t_span = torch.tensor([0.0, 5.0], device=device)
    traj1 = solve_ode_rk4(model.vector_field, z1_0, t_span, steps=25)
    traj2 = solve_ode_rk4(model.vector_field, z2_0, t_span, steps=25)
    
    final_dist = torch.norm(traj1[-1] - traj2[-1]).item()
    contraction_ratio = final_dist / max(1e-8, init_dist)
    
    model.train()
    return max_eig, init_dist, final_dist, contraction_ratio


def main():
    print("=" * 70)
    print("CLR MINIMAL FALSIFICATION LOOP: N-bit Parity + Approach A")
    print("=" * 70)
    
    set_seed(42)
    device = torch.device("cpu")
    seq_len = 16 # Start with 16-bit parity for rapid turnaround
    num_train = 1000
    num_val = 200
    batch_size = 32
    epochs = 10
    
    print(f"Task: {seq_len}-Bit Parity | Batch Size: {batch_size} | Training Samples: {num_train}")
    
    x_train_raw, y_train = generate_parity_dataset(num_samples=num_train, seq_len=seq_len, seed=42)
    x_val_raw, y_val = generate_parity_dataset(num_samples=num_val, seq_len=seq_len, seed=100)
    
    x_train = x_train_raw.unsqueeze(-1)
    x_val = x_val_raw.unsqueeze(-1)
    
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    
    model = MinimalContractiveReasoner(seq_len=seq_len, latent_dim=16, context_dim=32, hidden_dim=48).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    
    t_span = torch.tensor([0.0, 3.0], device=device)
    
    print("\nInitial State Contraction Check (Before Training):")
    max_eig, d0, d_final, ratio = run_online_contraction_check(model, x_val, device)
    print(f"  -> Max Eigenvalue λ_max(Sym(J)): {max_eig:.5f} (Contractive: {max_eig < 0})")
    print(f"  -> Initial dist: {d0:.4f}, Final dist (t=3.0): {d_final:.6f}, Contraction ratio: {ratio:.6e}")
    
    print("\nStarting Training & Tracking...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits, z_star = model(batch_x, t_span)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)
            
        train_acc = correct / total
        train_loss = total_loss / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                v_logits, _ = model(vx, t_span)
                v_preds = torch.argmax(v_logits, dim=-1)
                val_correct += (v_preds == vy).sum().item()
                val_total += vx.size(0)
        val_acc = val_correct / val_total
        
        # Online Contraction Diagnostic on TRAINED weights
        max_eig, d0, d_final, ratio = run_online_contraction_check(model, x_val, device)
        
        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.1f}% | "
              f"Val Acc: {val_acc*100:5.1f}% | λ_max: {max_eig:+.4f} | Dist Decay: {ratio:.2e}")
              
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f}s.")
    print("=" * 70)

    save_experiment_results(
        experiment_name="minimal_parity_loop",
        metrics={
            "final_train_loss": train_loss,
            "final_train_acc": train_acc,
            "final_val_acc": val_acc,
            "lambda_max_sym_j": max_eig,
            "distance_decay_ratio": ratio,
        },
        config={
            "seq_len": seq_len,
            "num_train": num_train,
            "num_val": num_val,
            "epochs": epochs,
            "seed": 42,
        },
        seed=42,
        start_time=start_time,
    )


if __name__ == "__main__":
    main()

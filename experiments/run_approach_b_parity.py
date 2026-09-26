"""
Approach B Falsification Experiment:
Unconstrained / Expressive Vector Field with Spectral Normalization and Contraction Regularization.

Objective:
Test whether removing the ICNN convexity restriction enables solving 8-bit parity
while maintaining Demidovich contraction: lambda_max(Sym(J)) <= -kappa < 0.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time

from data.generators.parity import generate_parity_dataset
from src.dynamics.solvers import solve_ode_rk4
from src.utils import set_seed, save_experiment_results


class ExpressiveVectorField(nn.Module):
    """
    General, non-potential vector field:
    f(z; c) = W_2 * tanh(W_1 * z + U * c + b_1) + b_2 - D * z
    
    Equipped with spectral normalization on W_1 and W_2 to bound the Lipschitz constant,
    combined with trainable positive damping D.
    """
    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        hidden_dim: int = 64,
        min_damping: float = 0.5,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.min_damping = min_damping
        
        # Spectrally normalized linear layers
        self.fc1_z = nn.utils.parametrizations.spectral_norm(nn.Linear(latent_dim, hidden_dim, bias=True))
        self.fc1_c = nn.Linear(context_dim, hidden_dim, bias=False)
        self.fc2 = nn.utils.parametrizations.spectral_norm(nn.Linear(hidden_dim, latent_dim, bias=True))
        
        # Trainable damping D
        self.log_d = nn.Parameter(torch.ones(latent_dim) * 0.0)
        self.context = None

    def set_context(self, c: torch.Tensor):
        self.context = c

    def get_damping(self) -> torch.Tensor:
        return F.softplus(self.log_d) + self.min_damping

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Non-conservative flow
        h = torch.tanh(self.fc1_z(z) + self.fc1_c(self.context))
        flow = self.fc2(h)
        
        # Damping force
        d = self.get_damping()
        dz_dt = flow - d * z
        return dz_dt


class ApproachBReasoner(nn.Module):
    def __init__(self, latent_dim: int = 24, context_dim: int = 32, hidden_dim: int = 64, min_damping: float = 0.5):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        
        # Encoder
        self.encoder = nn.GRU(
            input_size=1,
            hidden_size=context_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        
        # Expressive Vector Field
        self.vector_field = ExpressiveVectorField(
            latent_dim=latent_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            min_damping=min_damping,
        )
        
        # Readout Probe
        self.probe = nn.Linear(latent_dim, 2)

    def forward(self, x: torch.Tensor, t_span: torch.Tensor):
        batch_size = x.shape[0]
        _, h_n = self.encoder(x)
        c = torch.cat([h_n[0], h_n[1]], dim=-1)
        
        self.vector_field.set_context(c)
        z0 = torch.zeros(batch_size, self.latent_dim, device=x.device)
        
        traj = solve_ode_rk4(self.vector_field, z0, t_span, steps=25)
        z_star = traj[-1]
        logits = self.probe(z_star)
        return logits, z_star


def compute_spectral_contraction_stats(vector_field: ExpressiveVectorField, c_samples: torch.Tensor, latent_dim: int, device: torch.device):
    """
    Computes lambda_max(Sym(J)) and trajectory distance decay.
    """
    vector_field.eval()
    eps = 1e-4
    t0 = torch.tensor(0.0, device=device)
    
    max_eigs = []
    # Test across a batch of sample contexts and states
    batch_check = min(8, c_samples.size(0))
    z_test = torch.randn(batch_check, latent_dim, device=device)
    
    for i in range(batch_check):
        z_i = z_test[i:i+1]
        c_i = c_samples[i:i+1]
        vector_field.set_context(c_i)
        
        J = torch.zeros(latent_dim, latent_dim, device=device)
        for k in range(latent_dim):
            e_k = torch.zeros_like(z_i)
            e_k[0, k] = eps
            f_p = vector_field(t0, z_i + e_k)
            f_m = vector_field(t0, z_i - e_k)
            J[:, k] = ((f_p - f_m) / (2.0 * eps)).squeeze(0)
            
        sym_J = 0.5 * (J + J.T)
        eigs = torch.linalg.eigvalsh(sym_J)
        max_eigs.append(torch.max(eigs).item())
        
    global_max_eig = max(max_eigs)
    
    # Distance decay test
    z1_0 = torch.randn(1, latent_dim, device=device) * 2.0
    z2_0 = torch.randn(1, latent_dim, device=device) * 2.0
    d0 = torch.norm(z1_0 - z2_0).item()
    
    vector_field.set_context(c_samples[0:1])
    t_span = torch.tensor([0.0, 4.0], device=device)
    traj1 = solve_ode_rk4(vector_field, z1_0, t_span, steps=25)
    traj2 = solve_ode_rk4(vector_field, z2_0, t_span, steps=25)
    
    d_final = torch.norm(traj1[-1] - traj2[-1]).item()
    ratio = d_final / max(1e-8, d0)
    
    vector_field.train()
    return global_max_eig, ratio


def main():
    print("=" * 70)
    print("APPROACH B TEST: Expressive Vector Field + Spectral Norm on 8-Bit Parity")
    print("=" * 70)
    
    set_seed(42)
    device = torch.device("cpu")
    seq_len = 8
    num_train = 3000
    num_val = 600
    batch_size = 32
    epochs = 20
    
    print(f"Task: {seq_len}-Bit Parity | Min Damping: 0.5 | Epochs: {epochs}")
    
    x_train_raw, y_train = generate_parity_dataset(num_samples=num_train, seq_len=seq_len, seed=42)
    x_val_raw, y_val = generate_parity_dataset(num_samples=num_val, seq_len=seq_len, seed=100)
    
    x_train = x_train_raw.unsqueeze(-1)
    x_val = x_val_raw.unsqueeze(-1)
    
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    
    model = ApproachBReasoner(latent_dim=24, context_dim=32, hidden_dim=64, min_damping=0.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    t_span = torch.tensor([0.0, 3.0], device=device)
    
    # Pre-training check
    with torch.no_grad():
        _, h_n = model.encoder(x_val[:8])
        c_init = torch.cat([h_n[0], h_n[1]], dim=-1)
    max_eig, ratio = compute_spectral_contraction_stats(model.vector_field, c_init, model.latent_dim, device)
    print(f"\nInitial State: λ_max(Sym(J)) = {max_eig:+.4f} (Contractive: {max_eig < 0}) | Distance Decay: {ratio:.2e}")
    
    print("\nStarting Training...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits, z_star = model(bx, t_span)
            
            # Primary loss
            loss = criterion(logits, by)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * bx.size(0)
            correct += (logits.argmax(-1) == by).sum().item()
            total += bx.size(0)
            
        train_acc = correct / total
        train_loss = total_loss / total
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for vx, vy in val_loader:
                v_logits, _ = model(vx, t_span)
                val_correct += (v_logits.argmax(-1) == vy).sum().item()
                val_total += vx.size(0)
        val_acc = val_correct / val_total
        
        # Diagnostic on real weights
        with torch.no_grad():
            _, h_n = model.encoder(x_val[:8])
            c_val = torch.cat([h_n[0], h_n[1]], dim=-1)
        max_eig, ratio = compute_spectral_contraction_stats(model.vector_field, c_val, model.latent_dim, device)
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.1f}% | "
              f"Val Acc: {val_acc*100:5.1f}% | λ_max: {max_eig:+.4f} | Decay: {ratio:.2e}")
              
        if val_acc > 0.99:
            print("\n>>> SUCCESS: 8-Bit Parity SOLVED with Approach B! <<<")
            break
            
    elapsed = time.time() - start_time
    print(f"\nExecution finished in {elapsed:.2f}s.")
    print("=" * 70)

    save_experiment_results(
        experiment_name="approach_b_parity",
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

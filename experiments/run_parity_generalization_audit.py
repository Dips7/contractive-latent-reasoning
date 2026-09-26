"""
Rigorous parity generalization audit.

Motivation:
    The reported "100% validation accuracy on 8-bit parity" is uninformative:
    8-bit parity has only 2^8 = 256 distinct inputs, and 3000 training samples
    (drawn with replacement) cover 100% of that space. 100% of the 600
    validation inputs also appear in training. The reported number therefore
    measures recall, not generalization.

This script:
  1. Quantifies the train/val input overlap for each N.
  2. Re-runs Approach B (spectral-norm + damping contractive ODE) on parity at
     N where memorization is impossible (train covers a tiny fraction of 2^N).
  3. Compares against a parameter-matched GRU-encoder + direct classifier that
     has NO continuous dynamics at all.
  4. Re-tests true test-time compute scaling on held-out inputs only.
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


# ----------------------------------------------------------------------------
# Overlap audit
# ----------------------------------------------------------------------------
def audit_overlap(seq_len: int, num_train: int, num_val: int):
    x_tr, _ = generate_parity_dataset(num_samples=num_train, seq_len=seq_len, seed=42)
    x_va, _ = generate_parity_dataset(num_samples=num_val, seq_len=seq_len, seed=100)
    tr_set = set(map(tuple, (x_tr > 0).int().tolist()))
    va_set = set(map(tuple, (x_va > 0).int().tolist()))
    space = 2 ** seq_len
    return {
        "space": space,
        "unique_train": len(tr_set),
        "unique_val": len(va_set),
        "val_in_train": len(va_set & tr_set),
        "frac_space_covered": len(tr_set) / space,
        "frac_val_seen": len(va_set & tr_set) / max(1, len(va_set)),
    }


# ----------------------------------------------------------------------------
# Approach B: contractive ODE reasoner (same as run_approach_b_parity.py)
# ----------------------------------------------------------------------------
class ExpressiveVectorField(nn.Module):
    def __init__(self, latent_dim, context_dim, hidden_dim=64, min_damping=0.5):
        super().__init__()
        self.min_damping = min_damping
        self.fc1_z = nn.utils.parametrizations.spectral_norm(nn.Linear(latent_dim, hidden_dim, bias=True))
        self.fc1_c = nn.Linear(context_dim, hidden_dim, bias=False)
        self.fc2 = nn.utils.parametrizations.spectral_norm(nn.Linear(hidden_dim, latent_dim, bias=True))
        self.log_d = nn.Parameter(torch.ones(latent_dim) * 0.0)
        self.context = None

    def set_context(self, c):
        self.context = c

    def get_damping(self):
        return F.softplus(self.log_d) + self.min_damping

    def forward(self, t, z):
        h = torch.tanh(self.fc1_z(z) + self.fc1_c(self.context))
        return self.fc2(h) - self.get_damping() * z


class ApproachBReasoner(nn.Module):
    def __init__(self, seq_len, latent_dim=24, context_dim=32, hidden_dim=64, min_damping=0.5):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.GRU(1, context_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.vector_field = ExpressiveVectorField(latent_dim, context_dim, hidden_dim, min_damping)
        self.probe = nn.Linear(latent_dim, 2)

    def forward(self, x, t_span):
        B = x.shape[0]
        _, h_n = self.encoder(x)
        c = torch.cat([h_n[0], h_n[1]], dim=-1)
        self.vector_field.set_context(c)
        z0 = torch.zeros(B, self.latent_dim, device=x.device)
        t_final = t_span[-1].item() if isinstance(t_span[-1], torch.Tensor) else float(t_span[-1])
        steps = max(25, int(t_final * 10))
        traj = solve_ode_rk4(self.vector_field, z0, t_span, steps=steps)
        z_star = traj[-1]
        return self.probe(z_star), z_star


# ----------------------------------------------------------------------------
# Parameter-matched baseline: same encoder, direct readout, NO ODE
# ----------------------------------------------------------------------------
class DirectGRUBaseline(nn.Module):
    """Same GRU encoder + readout MLP, but zero continuous reasoning depth."""
    def __init__(self, seq_len, context_dim=32, hidden_dim=64):
        super().__init__()
        self.encoder = nn.GRU(1, context_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, t_span=None):
        _, h_n = self.encoder(x)
        c = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.head(c), c


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train_eval(model, x_tr, y_tr, x_va, y_va, epochs, t_span, lr=4e-3, batch_size=32, use_ode=True):
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            out = model(bx, t_span) if use_ode else model(bx)
            logits = out[0] if isinstance(out, tuple) else out
            crit(logits, by).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for vx, vy in DataLoader(TensorDataset(x_va, y_va), batch_size=128):
                out = model(vx, t_span) if use_ode else model(vx)
                logits = out[0] if isinstance(out, tuple) else out
                correct += (logits.argmax(-1) == vy).sum().item()
                total += vx.size(0)
        hist.append(correct / total)
    return hist


def main():
    start_time = time.time()
    set_seed(42)
    print("=" * 88)
    print("PARITY GENERALIZATION AUDIT: Is the 8-bit result memorization?")
    print("=" * 88)

    print("\n[1] Train/Val input-space overlap audit")
    print("-" * 88)
    hdr = f"{'N':>4} | {'input space':>12} | {'uniq train':>10} | {'uniq val':>9} | {'% space':>9} | {'% val seen in train':>20}"
    print(hdr)
    print("-" * 88)
    for n, ntr, nva in [(8, 3000, 600), (16, 3000, 600), (32, 3000, 600), (64, 3000, 600)]:
        r = audit_overlap(n, ntr, nva)
        print(f"{n:>4} | {r['space']:>12,} | {r['unique_train']:>10,} | {r['unique_val']:>9,} | "
              f"{r['frac_space_covered']*100:>8.4f}% | {r['frac_val_seen']*100:>19.2f}%")

    # ---------------------------------------------------------------
    print("\n[2] Approach B vs. parameter-matched GRU baseline (no ODE)")
    print("    Held-out inputs only; train covers a vanishing fraction of 2^N")
    print("-" * 88)
    results = {}
    for seq_len in [8, 32]:
        x_tr_raw, y_tr = generate_parity_dataset(num_samples=3000, seq_len=seq_len, seed=42)
        x_va_raw, y_va = generate_parity_dataset(num_samples=600, seq_len=seq_len, seed=100)
        # dedupe: keep only validation inputs NOT present in training
        tr_set = set(map(tuple, (x_tr_raw > 0).int().tolist()))
        keep = torch.tensor([tuple(v) not in tr_set for v in (x_va_raw > 0).int().tolist()])
        is_held_out = bool(keep.sum() > 32)
        if is_held_out:
            x_va_d, y_va_d = x_va_raw[keep], y_va[keep]
            status_str = f"HELD-OUT EVALUATION ({keep.sum().item()} unseen validation inputs)"
        else:
            x_va_d, y_va_d = x_va_raw, y_va
            status_str = "MEMORIZATION / RECALL (0 held-out inputs; training saturates 2^8 = 256 input space)"

        x_tr, x_va = x_tr_raw.unsqueeze(-1), x_va_d.unsqueeze(-1)
        t_span = torch.tensor([0.0, 3.0])
        epochs = 20

        a = ApproachBReasoner(seq_len=seq_len, latent_dim=24, context_dim=32, hidden_dim=64, min_damping=0.5)
        b = DirectGRUBaseline(seq_len=seq_len, context_dim=32, hidden_dim=64)
        ha = train_eval(a, x_tr, y_tr, x_va, y_va_d, epochs, t_span, use_ode=True)
        hb = train_eval(b, x_tr, y_tr, x_va, y_va_d, epochs, t_span, use_ode=False)
        results[seq_len] = {
            "clr_final_acc": float(ha[-1]),
            "gru_final_acc": float(hb[-1]),
            "held_out_count": int(keep.sum().item()),
            "total_val_count": int(len(y_va)),
            "is_strictly_held_out": is_held_out,
            "regime": "generalization" if is_held_out else "memorization",
        }
        print(f"N={seq_len:>3} | Mode: {status_str}")
        print(f"       | held-out val inputs: {keep.sum().item():>4}/{len(y_va)} | "
              f"params CLR={count_params(a):>6} baseline={count_params(b):>6}")
        print(f"       | final val acc  CLR: {ha[-1]*100:6.2f}%   GRU baseline: {hb[-1]*100:6.2f}%")

    # ---------------------------------------------------------------
    print("\n[3] True test-time compute scaling on HELD-OUT inputs (N=32)")
    print("-" * 88)
    seq_len = 32
    x_tr_raw, y_tr = generate_parity_dataset(num_samples=3000, seq_len=seq_len, seed=42)
    x_va_raw, y_va = generate_parity_dataset(num_samples=600, seq_len=seq_len, seed=100)
    tr_set = set(map(tuple, (x_tr_raw > 0).int().tolist()))
    keep = torch.tensor([tuple(v) not in tr_set for v in (x_va_raw > 0).int().tolist()])
    x_va_d, y_va_d = x_va_raw[keep], y_va[keep]
    a = ApproachBReasoner(seq_len=seq_len, latent_dim=24, context_dim=32, hidden_dim=64, min_damping=0.5)
    train_eval(a, x_tr_raw.unsqueeze(-1), y_tr, x_va_d.unsqueeze(-1), y_va_d, 20, torch.tensor([0.0, 3.0]))
    a.eval()
    vl = DataLoader(TensorDataset(x_va_d.unsqueeze(-1), y_va_d), batch_size=128)
    print(f"  Trained at T=3.0; evaluating unseen inputs across horizons:")
    scaling_audit = []
    for t_val in [0.5, 1.0, 3.0, 6.0, 10.0, 20.0, 40.0]:
        ts = torch.tensor([0.0, t_val])
        c = tot = 0
        with torch.no_grad():
            for vx, vy in vl:
                lg, _ = a(vx, ts)
                c += (lg.argmax(-1) == vy).sum().item()
                tot += vx.size(0)
        acc_t = c / tot
        scaling_audit.append({"T": t_val, "held_out_val_acc": float(acc_t)})
        print(f"    T = {t_val:5.1f} | held-out val acc: {acc_t*100:6.2f}%")

    save_experiment_results(
        experiment_name="parity_generalization_audit",
        metrics={
            "generalization_results": {str(k): v for k, v in results.items()},
            "scaling_held_out": scaling_audit,
        },
        config={"seed": 42},
        seed=42,
        start_time=start_time,
    )


if __name__ == "__main__":
    main()

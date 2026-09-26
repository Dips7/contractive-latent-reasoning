"""Regression tests documenting the critical audit findings.

Each test encodes a V&V finding that must not silently regress:

  1. test_row_normalized_adjacency_spectral_norm_can_exceed_one
     The row-normalized adjacency matrix does not guarantee ||A_norm||_2 <= 1.
     This test pins the empirical finding that ||A_norm||_2 > 1 can occur,
     clarifying that contraction relies on damping dominating the Lipschitz bound.

  2. test_empirical_contraction_holds_on_trained_graph_model
     Verifies that despite any loose analytical bounds, the trained graph reasoner
     field satisfies lambda_max(Sym(J)) < 0 along its trajectories.

  3. test_parity_val_inputs_are_memorized_at_N8
     At N=8, 3000 training samples saturate the 256-point input space, and 100%
     of validation inputs appear in training. Pins that N=8 evaluation is memorization/recall.

  4. test_solver_step_count_governs_stability
     Explicit fixed-step RK4 requires h * |lambda_max| < 2.78.
     A fixed 25 steps at T=200 diverges (h * d = 12 >> 2.78), whereas scaling step count
     with T keeps the solve stable and finite.
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dynamics.solvers import solve_ode_rk4


# ---------------------------------------------------------------------------
# 1. Spectral norm of row-normalized adjacency
# ---------------------------------------------------------------------------
def test_row_normalized_adjacency_spectral_norm_can_exceed_one():
    """||A_norm||_2 > 1 is common -> documented assumption boundary."""
    torch.manual_seed(0)
    found = False
    for _ in range(100):
        A = (torch.rand(16, 16) < 0.15).float()
        A.fill_diagonal_(0.0)
        A_norm = A / A.sum(-1, keepdim=True).clamp(min=1.0)
        if torch.linalg.svdvals(A_norm)[0].item() > 1.0 + 1e-9:
            found = True
            break
    assert found, "Expected to find random graphs where row-normalized ||A||_2 > 1"


# ---------------------------------------------------------------------------
# Small model copy used in tests 2 & 4
# ---------------------------------------------------------------------------
class _GraphCLR(nn.Module):
    """Minimal dynamics copy; mirrors the shipped graph reasoner vector field."""

    def __init__(self, num_nodes=8, node_dim=4, min_damping=1.5):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.min_damping = min_damping
        self.w1 = nn.utils.parametrizations.spectral_norm(nn.Linear(node_dim, node_dim, bias=False))
        self.w2 = nn.utils.parametrizations.spectral_norm(nn.Linear(node_dim, node_dim, bias=False))
        self.log_d = nn.Parameter(torch.zeros(node_dim))

    def get_damping(self):
        return F.softplus(self.log_d) + self.min_damping

    def vf(self, A_norm, src, z):
        """Vector field f(z) = W2 tanh(W1 (A_norm^T @ z)) - D z + src."""
        zp = torch.bmm(A_norm.transpose(1, 2), z)
        return self.w2(torch.tanh(self.w1(zp))) - self.get_damping() * z + src


# ---------------------------------------------------------------------------
# 2. Empirical contraction on a trained graph model
# ---------------------------------------------------------------------------
def test_empirical_contraction_holds_on_trained_graph_model():
    """Pin the empirical fact that the trained field is strictly contractive."""
    torch.manual_seed(0)
    n, d = 8, 4
    model = _GraphCLR(n, d, min_damping=1.5)
    head = nn.Linear(d, 2)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=8e-3)
    crit = nn.CrossEntropyLoss()
    y = torch.randint(0, 2, (64,))
    A = (torch.rand(64, n, n) < 0.2).float()
    A.diagonal(dim1=-2, dim2=-1).zero_()
    A_norm = A / A.sum(-1, keepdim=True).clamp(min=1.0)
    src = torch.zeros(64, n, d)
    src[:, 0, 0] = 1.0
    t_span = torch.tensor([0.0, 4.0])

    for _ in range(3):
        opt.zero_grad()
        z0 = src.clone()
        z = solve_ode_rk4(lambda t, z_in: model.vf(A_norm, src, z_in), z0, t_span, steps=25)[-1]
        logits = head(z[:, -1])
        crit(logits, y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()

    worst = -math.inf
    for bi in range(4):
        An = A_norm[bi]
        src_i = src[bi]
        z_flat = torch.zeros(n * d)
        dt = 4.0 / 25

        def f_eval(z_in_flat):
            zz = z_in_flat.view(n, d)
            zp = torch.mm(An.t(), zz)
            return (
                model.w2(torch.tanh(model.w1(zp)))
                - model.get_damping() * zz
                + src_i
            ).view(-1)

        for step in range(26):
            if step in (0, 8, 16, 25):
                J = torch.func.jacrev(f_eval)(z_flat)
                symJ = 0.5 * (J + J.transpose(0, 1))
                m = torch.linalg.eigvalsh(symJ).max().item()
                if m > worst:
                    worst = m
            with torch.no_grad():
                k1 = f_eval(z_flat)
                k2 = f_eval(z_flat + 0.5 * dt * k1)
                k3 = f_eval(z_flat + 0.5 * dt * k2)
                k4 = f_eval(z_flat + dt * k3)
                z_flat = z_flat + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    assert worst < 0, (
        f"Trained field not empirically contractive: "
        f"max lambda_max(Sym(J)) over 4 samples x 4 timepoints = {worst}"
    )


# ---------------------------------------------------------------------------
# 3. N=8 parity: val inputs ARE memorized
# ---------------------------------------------------------------------------
def test_parity_val_inputs_are_memorized_at_N8():
    """At N=8, 100% of validation inputs are also training inputs."""
    from data.generators.parity import generate_parity_dataset
    x_tr, _ = generate_parity_dataset(num_samples=3000, seq_len=8, seed=42)
    x_va, _ = generate_parity_dataset(num_samples=600, seq_len=8, seed=100)
    tr = set(map(tuple, (x_tr > 0).int().tolist()))
    va = set(map(tuple, (x_va > 0).int().tolist()))
    assert len(tr) == 256, "8-bit training set must saturate the 256-point input space"
    assert len(va & tr) == len(va), "Every N=8 validation input is also in training"


# ---------------------------------------------------------------------------
# 4. RK4 fixed-step instability at large T
# ---------------------------------------------------------------------------
def test_solver_step_count_governs_stability():
    """RK4 fixed steps diverge once h*d_max leaves the RK4 stability region."""
    d = 1.5
    model = _GraphCLR(4, 4, min_damping=d)
    model.eval()

    A = torch.eye(4).unsqueeze(0)
    src = torch.zeros(1, 4, 4)
    src[0, 0, 0] = 1.0
    A_norm = A / A.sum(-1, keepdim=True)
    vf = lambda t, z: model.vf(A_norm, src, z)

    # 25 fixed steps at T=200 -> h=8.0; h*d_max = 12.0 >> 2.78: must diverge
    z = solve_ode_rk4(vf, src, torch.tensor([0.0, 200.0]), steps=25)[-1]
    assert not torch.isfinite(z).all(), "Expected divergence at large T with fixed 25 steps"

    # Step count scaled with actual d_max: keeps h*d_max < 2.0 < 2.78
    d_max = model.get_damping().max().item()
    steps = int(200 * d_max / 1.5)
    z2 = solve_ode_rk4(vf, src, torch.tensor([0.0, 200.0]), steps=steps)[-1]
    assert torch.isfinite(z2).all(), "Expected finite state when step count scales with T"
    assert z2.norm() < 10.0, "State should be near the attractor"

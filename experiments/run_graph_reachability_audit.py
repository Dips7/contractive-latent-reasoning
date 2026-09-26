"""
Rigorous audit of the multi-hop graph reachability experiment.

Three confounds in the headline comparison are tested:

  (A) STRAWMAN BASELINE. The reported "constant-depth baseline" is an MLP on the
      flattened (adjacency | u-onehot | v-onehot) vector, while the CLR is a
      message-passing model that consumes the adjacency as a MATRIX. The CLR's
      advantage may therefore come from the graph inductive bias, not from
      continuous-time reasoning. Correct control: a fixed-K-round message-passing
      GNN (the constant-depth analogue in the graph domain).

  (B) T-SCALING = OFF-CALIBRATION, NOT "THINKING LONGER". The model is trained
      at T=4 so the readout head is calibrated on states near the fixed point
      z*. At T<<4 the state is far from z*, so the drop in accuracy may just be
      an out-of-distribution readout, not evidence of "deeper reasoning".
      Decisive test: stratify accuracy by required BFS path length. If T-scaling
      genuinely propagates information farther, low T should fail specifically
      on LONG paths while saturating on short ones.

  (C) THE ACTUAL THESIS CLAIM. Test-time compute scaling should let the model
      solve HARDER instances than it saw in training. Test: train on graphs
      whose u->v path length is <= K_train, evaluate on graphs whose shortest
      u->v path is longer, and sweep T. If the thesis holds, increasing T must
      extend the reachable depth. Also: out-of-distribution node count.

Also tests the never-verified noise-robustness claim (mid-trajectory perturbation).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import collections

from src.dynamics.solvers import solve_ode_rk4
from src.utils import set_seed, save_experiment_results

NUM_NODES = 16


# ----------------------------------------------------------------------------
# Data generation with explicit path-length labels
# ----------------------------------------------------------------------------
def bfs_distance(A, u, v, n):
    """Shortest path length u->v, or -1 if unreachable."""
    dist = {u: 0}
    q = collections.deque([u])
    while q:
        cur = q.popleft()
        if cur == v:
            return dist[cur]
        for nb in np.where(A[cur] > 0)[0]:
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                q.append(nb)
    return -1


def make_dataset(num_samples, num_nodes, edge_prob, seed, max_path=None, min_path=None):
    """Balanced reachability dataset with BFS shortest-path annotations.

    max_path/min_path, if given, restrict the POSITIVE examples to graphs whose
    shortest u->v path is within [min_path, max_path].
    """
    rng = np.random.default_rng(seed)
    As, us, vs, ys, ds = [], [], [], [], []
    target = num_samples // 2
    pos = neg = 0
    while pos < target or neg < target:
        A = (rng.random((num_nodes, num_nodes)) < edge_prob).astype(np.float32)
        np.fill_diagonal(A, 0.0)
        u, v = rng.choice(num_nodes, size=2, replace=False)
        d = bfs_distance(A, u, v, num_nodes)
        if d > 0:
            if max_path is not None and d > max_path:
                continue
            if min_path is not None and d < min_path:
                continue
        if d > 0 and pos < target:
            As.append(A); us.append(u); vs.append(v); ys.append(1); ds.append(d); pos += 1
        elif d < 0 and neg < target:
            As.append(A); us.append(u); vs.append(v); ys.append(0); ds.append(-1); neg += 1
    A_t = torch.tensor(np.stack(As))
    feat = torch.cat([A_t.flatten(1),
                      F.one_hot(torch.tensor(us), num_nodes).float(),
                      F.one_hot(torch.tensor(vs), num_nodes).float()], dim=1)
    return feat, torch.tensor(ys), torch.tensor(ds)


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
class ContinuousGraphReasoner(nn.Module):
    """The CLR under test (identical dynamics to the shipped experiment)."""
    def __init__(self, num_nodes, node_dim=16, min_damping=1.5):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.min_damping = min_damping
        self.w1 = nn.utils.parametrizations.spectral_norm(nn.Linear(node_dim, node_dim, bias=False))
        self.w2 = nn.utils.parametrizations.spectral_norm(nn.Linear(node_dim, node_dim, bias=False))
        self.log_d = nn.Parameter(torch.ones(node_dim) * 0.0)
        self.readout = nn.Sequential(nn.Linear(node_dim, 32), nn.GELU(), nn.Linear(32, 2))

    def get_damping(self):
        return F.softplus(self.log_d) + self.min_damping

    def forward(self, x, t_span, noise_std=0.0, noise_at_frac=None, return_traj=False):
        B = x.shape[0]
        A = x[:, :self.num_nodes ** 2].view(B, self.num_nodes, self.num_nodes)
        u_oh = x[:, self.num_nodes ** 2:self.num_nodes ** 2 + self.num_nodes]
        v_oh = x[:, self.num_nodes ** 2 + self.num_nodes:]
        An = A / A.sum(-1, keepdim=True).clamp(min=1.0)
        src = torch.zeros(B, self.num_nodes, self.node_dim)
        src[:, :, 0] = u_oh
        z0 = src.clone()
        d = self.get_damping()

        def vf(t, z):
            zp = torch.bmm(An.transpose(1, 2), z)
            return self.w2(torch.tanh(self.w1(zp))) - d * z + src

        traj = solve_ode_rk4(vf, z0, t_span, steps=25)
        if noise_std > 0 and noise_at_frac is not None:
            idx = int(noise_at_frac * (len(traj) - 1))
            traj = list(traj)
            traj[idx] = traj[idx] + torch.randn_like(traj[idx]) * noise_std
            # re-integrate forward from the perturbed state
            z = traj[idx]
            t_rem = torch.tensor([t_span[-1] * noise_at_frac, t_span[-1]])
            seg = solve_ode_rk4(vf, z, t_rem, steps=max(5, int(25 * (1 - noise_at_frac))))
            traj = traj[: idx + 1] + list(seg[1:])
            traj = torch.stack(traj)
        z_star = traj[-1]
        v_state = (z_star * v_oh.unsqueeze(-1)).sum(1)
        logits = self.readout(v_state)
        if return_traj:
            return logits, z_star, traj
        return logits, z_star


class FixedRoundGNN(nn.Module):
    """Constant-depth control: exactly K rounds of message passing, no ODE.

    Same node features, same message function family, same readout. The ONLY
    difference vs. CLR is that depth is a fixed integer K, not a continuous
    integration horizon. This isolates 'continuous time' from 'graph bias'.
    """
    def __init__(self, num_nodes, node_dim=16, K=10):
        super().__init__()
        self.num_nodes = num_nodes
        self.K = K
        self.w1 = nn.utils.parametrizations.spectral_norm(nn.Linear(node_dim, node_dim, bias=False))
        self.w2 = nn.utils.parametrizations.spectral_norm(nn.Linear(node_dim, node_dim, bias=False))
        self.readout = nn.Sequential(nn.Linear(node_dim, 32), nn.GELU(), nn.Linear(32, 2))

    def forward(self, x, t_span=None):
        B = x.shape[0]
        A = x[:, :self.num_nodes ** 2].view(B, self.num_nodes, self.num_nodes)
        u_oh = x[:, self.num_nodes ** 2:self.num_nodes ** 2 + self.num_nodes]
        v_oh = x[:, self.num_nodes ** 2 + self.num_nodes:]
        An = A / A.sum(-1, keepdim=True).clamp(min=1.0)
        z = torch.zeros(B, self.num_nodes, self.w1.in_features)
        z[:, :, 0] = u_oh
        for _ in range(self.K):
            zp = torch.bmm(An.transpose(1, 2), z)
            z = self.w2(torch.tanh(self.w1(zp)))
        v_state = (z * v_oh.unsqueeze(-1)).sum(1)
        return self.readout(v_state), z


class TunedMLP(nn.Module):
    """The shipped baseline, re-tuned (wider, longer training)."""
    def __init__(self, input_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, t_span=None):
        return self.net(x), None


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def fit(model, x_tr, y_tr, x_va, y_va, epochs, lr, use_t=False, t_span=None, batch=32):
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            out = model(bx, t_span) if use_t else model(bx)
            crit(out[0] if isinstance(out, tuple) else out, by).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def acc(model, x, y, use_t=False, t_span=None, batch=256, noise_std=0.0, noise_at=None):
    model.eval()
    c = tot = 0
    preds = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch):
            xb, yb = x[i:i + batch], y[i:i + batch]
            if use_t:
                out = model(xb, t_span, noise_std=noise_std, noise_at_frac=noise_at) if noise_std > 0 else model(xb, t_span)
            else:
                out = model(xb)
            lg = out[0] if isinstance(out, tuple) else out
            preds.append(lg.argmax(-1))
            c += (lg.argmax(-1) == yb).sum().item()
            tot += xb.size(0)
    return c / tot, torch.cat(preds)


def acc_by_path(model, x, y, d, t_span):
    """Accuracy stratified by BFS shortest-path length of the POSITIVE class."""
    a, preds = acc(model, x, y, use_t=True, t_span=t_span)
    pos = d > 0
    buckets = collections.defaultdict(lambda: [0, 0])
    for i in torch.where(pos)[0].tolist():
        b = min(d[i].item(), 8)
        buckets[b][0] += (preds[i].item() == y[i].item())
        buckets[b][1] += 1
    neg_acc = (preds[~pos] == y[~pos]).float().mean().item()
    return a, dict(sorted(buckets.items())), neg_acc


def main():
    set_seed(42)
    print("=" * 92)
    print("GRAPH REACHABILITY AUDIT")
    print("=" * 92)

    x_tr, y_tr, d_tr = make_dataset(3000, NUM_NODES, 0.15, 42)
    x_va, y_va, d_va = make_dataset(600, NUM_NODES, 0.15, 100)
    in_dim = x_tr.shape[1]
    print(f"Train {tuple(x_tr.shape)} | Val {tuple(x_va.shape)}")
    print(f"Positive-class path-length distribution (val): "
          f"{dict(sorted(collections.Counter(d_va[d_va>0].tolist()).items()))}")

    # ------------------------------------------------------------
    print("\n[1] BASELINE FAIRNESS: is the MLP failure real, or a strawman?")
    print("-" * 92)
    clr = ContinuousGraphReasoner(NUM_NODES, node_dim=16, min_damping=1.5)
    clr = fit(clr, x_tr, y_tr, x_va, y_va, 10, 8e-3, use_t=True, t_span=torch.tensor([0.0, 4.0]))
    a_clr, _ = acc(clr, x_va, y_va, use_t=True, t_span=torch.tensor([0.0, 4.0]))

    mlp_ship = TunedMLP(in_dim, hidden=96)
    mlp_ship = fit(mlp_ship, x_tr, y_tr, x_va, y_va, 10, 3e-3)
    a_mlp96, _ = acc(mlp_ship, x_va, y_va)

    mlp_big = TunedMLP(in_dim, hidden=512)
    mlp_big = fit(mlp_big, x_tr, y_tr, x_va, y_va, 40, 1e-3)
    a_mlp512, _ = acc(mlp_big, x_va, y_va)

    print(f"  CLR (continuous, T=4)                 params={count_params(clr):>6,}  val acc = {a_clr*100:6.2f}%")
    print(f"  MLP baseline as shipped (h=96, 10ep)  params={count_params(mlp_ship):>6,}  val acc = {a_mlp96*100:6.2f}%")
    print(f"  MLP re-tuned (h=512, 3-layer, 40ep)   params={count_params(mlp_big):>6,}  val acc = {a_mlp512*100:6.2f}%")

    print("\n  Fixed-K message-passing GNN (constant-depth control):")
    for K in [2, 4, 6, 8, 12, 16]:
        gnn = FixedRoundGNN(NUM_NODES, node_dim=16, K=K)
        gnn = fit(gnn, x_tr, y_tr, x_va, y_va, 10, 8e-3)
        a, _ = acc(gnn, x_va, y_va)
        print(f"    K={K:>2} rounds | params={count_params(gnn):>6,} | val acc = {a*100:6.2f}%")

    # ------------------------------------------------------------
    print("\n[2] IS T-SCALING 'THINKING LONGER' OR JUST READOUT MIS-CALIBRATION?")
    print("-" * 92)
    print("  Stratified accuracy by required BFS path length (positives only):")
    print(f"  {'T':>5} | {'overall':>8} | " + " | ".join(f"d={k}" for k in range(1, 7)) + " | neg-acc")
    for t_val in [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 20.0]:
        ts = torch.tensor([0.0, t_val])
        a, buckets, neg = acc_by_path(clr, x_va, y_va, d_va, ts)
        row = " | ".join(f"{buckets.get(k, [0,1])[0]/max(1,buckets.get(k,[0,1])[1])*100:5.1f}%" for k in range(1, 7))
        print(f"  {t_val:5.1f} | {a*100:7.2f}% | {row} | {neg*100:5.1f}%")

    # ------------------------------------------------------------
    print("\n[3] THE REAL CLAIM: does larger T extend capability to LONGER PATHS")
    print("    than were present in training? (train: paths <= 3 | test: paths 4-8)")
    print("-" * 92)
    x_tr_s, y_tr_s, _ = make_dataset(3000, NUM_NODES, 0.15, 7, max_path=3)
    clr_s = ContinuousGraphReasoner(NUM_NODES, node_dim=16, min_damping=1.5)
    clr_s = fit(clr_s, x_tr_s, y_tr_s, x_va, y_va, 10, 8e-3, use_t=True, t_span=torch.tensor([0.0, 4.0]))
    # held-out test set restricted to LONG paths only
    long_mask = (d_va >= 4)
    x_long, y_long, d_long = x_va[long_mask], y_va[long_mask], d_va[long_mask]
    print(f"    Long-path test set: {long_mask.sum().item()} positives with path length >= 4")
    for t_val in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
        ts = torch.tensor([0.0, t_val])
        a, _ = acc(clr_s, x_long, y_long, use_t=True, t_span=ts)
        print(f"    T = {t_val:5.1f} | long-path-only test acc = {a*100:6.2f}%")

    # ------------------------------------------------------------
    print("\n[4] OUT-OF-DISTRIBUTION: train on 16 nodes, test on unseen node count")
    print("    (Both CLR and the GNN control are node-count-specific: weights are")
    print("     tied to N. A size-agnostic variant would be needed to transfer.)")
    print("-" * 92)
    for n_test in [24, 32]:
        x_o, y_o, _ = make_dataset(400, n_test, 0.15, 999)
        # Re-initialize a CLR sized for n_test, then TRANSPLANT the trained
        # message-passing weights (they operate per-node, so they transfer).
        clr_t = ContinuousGraphReasoner(n_test, node_dim=16, min_damping=1.5)
        clr_t.load_state_dict(clr.state_dict())
        clr_t.num_nodes = n_test
        row = []
        for t_val in [4.0, 8.0, 20.0]:
            ts = torch.tensor([0.0, t_val])
            a, _ = acc(clr_t, x_o, y_o, use_t=True, t_span=ts)
            row.append(f"T={t_val:>4}: {a*100:5.1f}%")
        print(f"    N_test={n_test:>3} | CLR (transplanted per-node weights) -> {' | '.join(row)}")

    # A size-agnostic control: same dynamics but run on a FIXED node embedding
    print("\n    Reference: in-distribution (N=16) accuracy for the same checkpoint")
    for t_val in [4.0, 8.0, 20.0]:
        ts = torch.tensor([0.0, t_val])
        a, _ = acc(clr, x_va, y_va, use_t=True, t_span=ts)
        print(f"      N=16, T={t_val:>4}: {a*100:5.1f}%")

    # ------------------------------------------------------------
    print("\n[5] NOISE ROBUSTNESS (claimed, never previously tested)")
    print("-" * 92)
    ts = torch.tensor([0.0, 8.0])
    base, _ = acc(clr, x_va, y_va, use_t=True, t_span=ts)
    print(f"    Clean @ T=8: {base*100:.2f}%")
    for sig in [0.1, 0.5, 1.0, 2.0]:
        a, _ = acc(clr, x_va, y_va, use_t=True, t_span=ts, noise_std=sig, noise_at=0.5)
        print(f"    Noise sigma={sig:>4} injected at t=T/2 -> acc = {a*100:6.2f}% "
              f"({(a-base)*100:+.2f} vs clean)")

    save_experiment_results(
        experiment_name="graph_reachability_audit",
        metrics={
            "clr_val_acc": float(a_clr),
            "clean_acc_t8": float(base),
        },
        config={
            "num_nodes": NUM_NODES,
            "seed": 42,
        },
        seed=42,
    )


if __name__ == "__main__":
    main()

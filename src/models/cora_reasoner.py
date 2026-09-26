"""Model architectures for Real-World Cora Reachability Reasoning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from src.dynamics.solvers import solve_ode_rk4


class CoraDirectBaseline(nn.Module):
    """
    Direct Embedding + MLP Baseline (No iterative graph depth).
    Maps source and target paper IDs into embedding space and classifies directly.
    """
    def __init__(self, num_nodes: int = 2708, embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.node_embed = nn.Embedding(num_nodes, embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, sources: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        u_emb = self.node_embed(sources)
        v_emb = self.node_embed(targets)
        feat = torch.cat([u_emb, v_emb], dim=-1)
        return self.classifier(feat)


class CoraDiscreteGNN(nn.Module):
    """
    Discrete Message Passing GNN with fixed K layers.
    Exhibits the fundamental horizon truncation problem: Cannot propagate beyond K hops.
    """
    def __init__(self, num_nodes: int = 2708, node_dim: int = 16, num_layers: int = 2):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([
            nn.Linear(node_dim, node_dim, bias=False) for _ in range(num_layers)
        ])
        self.readout = nn.Sequential(
            nn.Linear(node_dim + 1, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )

    def forward(self, A_sparse: torch.Tensor, sources: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B = sources.shape[0]
        N = self.num_nodes
        d = self.node_dim
        device = sources.device
        
        # Initial impulse at source node
        h = torch.zeros(B, N, d, device=device)
        for b in range(B):
            h[b, sources[b], 0] = 1.0
            
        for layer in self.layers:
            # Batched sparse propagation: A_sparse is (N, N), h is (B, N, d)
            h_flat = h.permute(1, 0, 2).reshape(N, B * d)
            prop_flat = torch.sparse.mm(A_sparse, h_flat)
            prop = prop_flat.view(N, B, d).permute(1, 0, 2)
            h = F.relu(layer(prop))
            
        # Target node readout with logarithmic dynamic range awareness
        target_states = torch.stack([h[b, targets[b]] for b in range(B)], dim=0)
        norms = torch.log(torch.norm(target_states, dim=-1, keepdim=True) + 1e-12)
        feat = torch.cat([target_states, norms], dim=-1)
        return self.readout(feat)


class CoraContractiveReasoner(nn.Module):
    """
    Continuous Contractive Latent Dynamical Reasoner (CLR) on Cora.
    Evolves latent state z(t) over continuous time horizon T:
        dz/dt = W_2 * tanh(A_norm^T * z * W_1) - D * z + S_u
    Contraction guaranteed by:
        lambda_max(Sym(J)) <= ||A_norm||_2 * ||W_1||_2 * ||W_2||_2 - d_min <= 1.0 - d_min < 0
    """
    def __init__(
        self,
        num_nodes: int = 2708,
        node_dim: int = 16,
        min_damping: float = 1.5,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.min_damping = min_damping
        
        # Spectrally normalized transforms guaranteeing ||W1||_2 <= 1, ||W2||_2 <= 1
        self.w1 = nn.utils.parametrizations.spectral_norm(nn.Linear(node_dim, node_dim, bias=False))
        self.w2 = nn.utils.parametrizations.spectral_norm(nn.Linear(node_dim, node_dim, bias=False))
        
        # Learnable damping D >= min_damping
        self.log_d = nn.Parameter(torch.ones(node_dim) * 0.0)
        
        # Readout classifier with logarithmic dynamic range awareness
        self.readout = nn.Sequential(
            nn.Linear(node_dim + 1, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )

    def get_damping(self) -> torch.Tensor:
        return F.softplus(self.log_d) + self.min_damping

    def forward(
        self,
        A_sparse: torch.Tensor,
        sources: torch.Tensor,
        targets: torch.Tensor,
        t_span: Optional[torch.Tensor] = None,
        perturbation_std: float = 0.0,
        perturbation_time: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrates continuous reasoning flow from t_span[0] to t_span[-1].
        Optionally injects perturbation at perturbation_time to test self-healing.
        """
        B = sources.shape[0]
        N = self.num_nodes
        d = self.node_dim
        device = sources.device
        
        if t_span is None:
            t_span = torch.tensor([0.0, 4.0], device=device)
            
        # Continuous source injection at source paper u
        source = torch.zeros(B, N, d, device=device)
        for b in range(B):
            source[b, sources[b], 0] = 1.0
        z0 = source.clone()
        
        damping = self.get_damping()
        
        def vf(t, z):
            # Sparse propagation along directed citation edges
            z_flat = z.permute(1, 0, 2).reshape(N, B * d)
            prop_flat = torch.sparse.mm(A_sparse, z_flat)
            prop = prop_flat.view(N, B, d).permute(1, 0, 2)
            flow = self.w2(torch.tanh(self.w1(prop)))
            return flow - damping * z + source
            
        t_final = t_span[-1].item() if isinstance(t_span[-1], torch.Tensor) else float(t_span[-1])
        steps = max(15, int(t_final * 6))
        
        if perturbation_time is not None:
            # Integrate in two stages: [0, t_pert] -> optionally inject noise -> [t_pert, t_final]
            # Uses identical numerical integration discretization across both arms (sigma == 0 and sigma > 0)
            t_mid = float(perturbation_time)
            steps_1 = max(8, int(t_mid * 6))
            steps_2 = max(8, int((t_final - t_mid) * 6))
            
            t_span_1 = torch.tensor([0.0, t_mid], device=device)
            traj_1 = solve_ode_rk4(vf, z0, t_span_1, steps=steps_1)
            z_mid = traj_1[-1]
            
            if perturbation_std > 0.0:
                noise = torch.randn_like(z_mid) * perturbation_std
                z_perturbed = z_mid + noise
            else:
                z_perturbed = z_mid
                
            t_span_2 = torch.tensor([t_mid, t_final], device=device)
            traj_2 = solve_ode_rk4(vf, z_perturbed, t_span_2, steps=steps_2)
            z_star = traj_2[-1]
        else:
            traj = solve_ode_rk4(vf, z0, t_span, steps=steps)
            z_star = traj[-1]
            
        # Readout at target paper v
        target_states = torch.stack([z_star[b, targets[b]] for b in range(B)], dim=0)
        norms = torch.log(torch.norm(target_states, dim=-1, keepdim=True) + 1e-12)
        feat = torch.cat([target_states, norms], dim=-1)
        logits = self.readout(feat)
        return logits, z_star


def compute_empirical_cora_sym_j(
    model: CoraContractiveReasoner,
    A_sparse: torch.Tensor,
    z_eval: Optional[torch.Tensor] = None,
    num_iters: int = 15,
    c: float = 10.0,
    eps: float = 1e-5,
) -> float:
    """
    Computes empirical lambda_max(Sym(J)) of the continuous Cora CLR vector field
    across the full 43,328-dimensional state space (2708 papers * 16 dimensions).
    
    Uses shifted power iteration:
        M_shifted = Sym(J) + c * I
    where J v is evaluated via central differences and J^T v is evaluated via autograd VJP.
    Because PyTorch forward-AD lacks sparse addmm support, this hybrid operator
    efficiently computes Sym(J) v in O(E * d) per iteration without dense Jacobian materialization.
    """
    model.eval()
    N = model.num_nodes
    d = model.node_dim
    device = next(model.parameters()).device
    
    if z_eval is None:
        # Default evaluation at source impulse fixed point
        s_dummy = torch.tensor([0], device=device)
        t_dummy = torch.tensor([1], device=device)
        t_span = torch.tensor([0.0, 4.0], device=device)
        with torch.no_grad():
            _, z_eval = model(A_sparse, s_dummy, t_dummy, t_span=t_span)
            
    damping = model.get_damping()
    
    def f_vec(z):
        z_flat = z.permute(1, 0, 2).reshape(N, d)
        prop_flat = torch.sparse.mm(A_sparse, z_flat)
        prop = prop_flat.view(N, 1, d).permute(1, 0, 2)
        flow = model.w2(torch.tanh(model.w1(prop)))
        return flow - damping * z

    z = z_eval.clone().detach().requires_grad_(True)
    out = f_vec(z)
    
    v = torch.randn_like(z)
    v = v / torch.norm(v)
    
    for _ in range(num_iters):
        with torch.no_grad():
            jv = (f_vec(z + eps * v) - f_vec(z - eps * v)) / (2.0 * eps)
        jt_v = torch.autograd.grad(out, z, grad_outputs=v, retain_graph=True)[0]
        sym_v = 0.5 * (jv + jt_v)
        v_next = sym_v + c * v
        v = v_next / torch.norm(v_next)
        
    with torch.no_grad():
        jv = (f_vec(z + eps * v) - f_vec(z - eps * v)) / (2.0 * eps)
    rayleigh = torch.sum(v * jv).item()
    return float(rayleigh)


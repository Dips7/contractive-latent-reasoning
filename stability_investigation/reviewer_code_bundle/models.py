"""
Consolidated Model Architectures for Reviewer Replication:
- ContinuousGraphODE (Single-scale contractive Neural ODE with Demidovich bounds)
- MultiscaleContractiveODE (Candidate B: multiscale skip-hop P and P^2)
- StructuralReadoutODE (Candidate C: exploratory degree-augmented readout)
- DiscreteGNN (6-layer discrete message-passing baseline)
- Numerical solvers (explicit_rk4_step) and SVD projection (bounded_weight)
"""

import math
from typing import Tuple, Optional, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

def bounded_weight(weight: torch.Tensor) -> torch.Tensor:
    """Clamps maximum singular value to <= 1.0."""
    sigma = torch.linalg.svdvals(weight)[0]
    return weight / torch.clamp(sigma * (1.0 + 1e-5), min=1.0)

def explicit_rk4_step(vf, t, z, dt):
    k1 = vf(t, z)
    k2 = vf(t + 0.5 * dt, z + 0.5 * dt * k1)
    k3 = vf(t + 0.5 * dt, z + 0.5 * dt * k2)
    k4 = vf(t + dt, z + dt * k3)
    return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

class DiscreteGNN(nn.Module):
    """K-layer discrete message passing GNN."""
    def __init__(self, node_dim: int = 8, num_layers: int = 6, readout_type: str = "log_norm"):
        super().__init__()
        self.node_dim = node_dim
        self.num_layers = num_layers
        self.readout_type = readout_type
        
        self.layers = nn.ModuleList([
            nn.Linear(node_dim, node_dim, bias=False) for _ in range(num_layers)
        ])
        in_dim = (node_dim + 1) if readout_type == "log_norm" else (2 * node_dim + 1)
        self.readout = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2)
        )

    def forward(self, A_sparse: torch.Tensor, sources: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B = sources.shape[0]
        N = A_sparse.shape[0]
        d = self.node_dim
        device = sources.device
        
        h = torch.zeros(B, N, d, device=device)
        h[torch.arange(B, device=device), sources, 0] = 1.0
        
        for layer in self.layers:
            # Flatten across batch: (N, B, d) -> (N, B*d)
            h_perm = h.permute(1, 0, 2).reshape(N, -1)
            prop = torch.sparse.mm(A_sparse, h_perm)
            prop = prop.reshape(N, B, d).permute(1, 0, 2)
            h = F.relu(layer(prop)) + h
            
        t_states = h[torch.arange(B, device=device), targets]
        if self.readout_type == "log_norm":
            norm = torch.linalg.vector_norm(t_states, dim=-1, keepdim=True)
            feat = torch.cat([t_states, torch.log(norm + 1e-12)], dim=-1)
        else:
            s_states = h[torch.arange(B, device=device), sources]
            cos = (s_states * t_states).sum(dim=-1, keepdim=True)
            feat = torch.cat([s_states, t_states, cos], dim=-1)
            
        return self.readout(feat)

class ContinuousGraphODE(nn.Module):
    """
    Continuous Graph Neural ODE with verifiable operator-norm contraction.
    Supports arbitrary graph sizes, damping parameter sweep, and readout controls.
    """
    def __init__(
        self,
        node_dim: int = 8,
        damping: bool = True,
        damping_offset: float = 1.5,
        method: str = "rk4",
        steps: int = 16,
        horizon: float = 4.0,
        readout_type: str = "log_norm",
    ):
        super().__init__()
        self.node_dim = node_dim
        self.damped = damping
        self.damping_offset = damping_offset
        self.method = method
        self.steps = steps
        self.horizon = horizon
        self.readout_type = readout_type
        
        self.w1 = nn.Linear(node_dim, node_dim, bias=False)
        self.w2 = nn.Linear(node_dim, node_dim, bias=False)
        self.log_d = nn.Parameter(torch.zeros(node_dim), requires_grad=damping)
        
        in_dim = (node_dim + 1) if readout_type == "log_norm" else (2 * node_dim + 1)
        self.readout = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2)
        )

    def field_parameters(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w1 = bounded_weight(self.w1.weight)
        w2 = bounded_weight(self.w2.weight)
        if self.damped:
            d = F.softplus(self.log_d) + self.damping_offset
        else:
            d = torch.zeros_like(self.log_d)
        return w1, w2, d

    def vector_field(self, A_sparse: torch.Tensor, source: torch.Tensor, params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        w1, w2, d = params
        N = A_sparse.shape[0]
        def field(t, z):
            B = z.shape[0]
            z_flat = z.permute(1, 0, 2).reshape(N, -1)
            prop = torch.sparse.mm(A_sparse, z_flat)
            prop = prop.reshape(N, B, self.node_dim).permute(1, 0, 2)
            act = torch.tanh(F.linear(prop, w1))
            return F.linear(act, w2) - d * z + source
        return field

    def forward(self, A_sparse: torch.Tensor, sources: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B = sources.shape[0]
        N = A_sparse.shape[0]
        d = self.node_dim
        device = sources.device
        
        source = self.w1.weight.new_zeros((B, N, d))
        source[torch.arange(B, device=device), sources, 0] = 1.0
        
        params = self.field_parameters()
        field = self.vector_field(A_sparse, source, params)
        
        z = source.clone()
        dt = self.horizon / self.steps
        for k in range(self.steps):
            if self.method == "rk4":
                z = explicit_rk4_step(field, k * dt, z, dt)
            else:
                z = z + dt * field(k * dt, z)
                
        t_states = z[torch.arange(B, device=device), targets]
        if self.readout_type == "log_norm":
            norm = torch.linalg.vector_norm(t_states, dim=-1, keepdim=True)
            feat = torch.cat([t_states, torch.log(norm + 1e-12)], dim=-1)
        else:
            s_states = z[torch.arange(B, device=device), sources]
            cos = (s_states * t_states).sum(dim=-1, keepdim=True)
            feat = torch.cat([s_states, t_states, cos], dim=-1)
            
        return self.readout(feat)

class MultiscaleContractiveODE(nn.Module):
    """
    Candidate B: Multiscale Skip-Hop Continuous Graph ODE.
    Vector field:
      dZ/dt = gamma1 * tanh(P1 Z B11) B12 + gamma2 * tanh(P2 Z B21) B22 - Z D + X
    where P1 = P, P2 = P^2.
    Certified Demidovich contraction margin:
      alpha = d_min - (gamma1 * ||B11||*||B12|| + gamma2 * ||B21||*||B22||) >= d_min - 1.0.
    """
    def __init__(
        self,
        node_dim: int = 8,
        gamma1: float = 0.6,
        gamma2: float = 0.4,
        damping_offset: float = 1.5,
        method: str = "rk4",
        steps: int = 16,
        horizon: float = 4.0,
        readout_type: str = "log_norm",
    ):
        super().__init__()
        assert abs((gamma1 + gamma2) - 1.0) < 1e-5, "Gammas must sum to 1.0"
        self.node_dim = node_dim
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.damping_offset = damping_offset
        self.method = method
        self.steps = steps
        self.horizon = horizon
        self.readout_type = readout_type
        
        # Scale 1 weights (1-hop)
        self.w11 = nn.Linear(node_dim, node_dim, bias=False)
        self.w12 = nn.Linear(node_dim, node_dim, bias=False)
        
        # Scale 2 weights (2-hop skip)
        self.w21 = nn.Linear(node_dim, node_dim, bias=False)
        self.w22 = nn.Linear(node_dim, node_dim, bias=False)
        
        self.log_d = nn.Parameter(torch.zeros(node_dim), requires_grad=True)
        
        in_dim = (node_dim + 1) if readout_type == "log_norm" else (2 * node_dim + 1)
        self.readout = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2)
        )

    def field_parameters(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        w11 = bounded_weight(self.w11.weight)
        w12 = bounded_weight(self.w12.weight)
        w21 = bounded_weight(self.w21.weight)
        w22 = bounded_weight(self.w22.weight)
        d = F.softplus(self.log_d) + self.damping_offset
        return w11, w12, w21, w22, d

    def vector_field(self, A_sparse: torch.Tensor, source: torch.Tensor, params):
        w11, w12, w21, w22, d = params
        N = A_sparse.shape[0]
        g1, g2 = self.gamma1, self.gamma2
        
        def field(t, z):
            B = z.shape[0]
            z_flat = z.permute(1, 0, 2).reshape(N, -1)
            
            # Scale 1: P z
            prop1 = torch.sparse.mm(A_sparse, z_flat)
            # Scale 2: P (P z) = P^2 z
            prop2 = torch.sparse.mm(A_sparse, prop1)
            
            p1_reshaped = prop1.reshape(N, B, self.node_dim).permute(1, 0, 2)
            p2_reshaped = prop2.reshape(N, B, self.node_dim).permute(1, 0, 2)
            
            term1 = g1 * F.linear(torch.tanh(F.linear(p1_reshaped, w11)), w12)
            term2 = g2 * F.linear(torch.tanh(F.linear(p2_reshaped, w21)), w22)
            
            return term1 + term2 - d * z + source
        return field

    def forward(self, A_sparse: torch.Tensor, sources: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B = sources.shape[0]
        N = A_sparse.shape[0]
        d = self.node_dim
        device = sources.device
        
        source = self.w11.weight.new_zeros((B, N, d))
        source[torch.arange(B, device=device), sources, 0] = 1.0
        
        params = self.field_parameters()
        field = self.vector_field(A_sparse, source, params)
        
        z = source.clone()
        dt = self.horizon / self.steps
        for k in range(self.steps):
            if self.method == "rk4":
                z = explicit_rk4_step(field, k * dt, z, dt)
            else:
                z = z + dt * field(k * dt, z)
                
        t_states = z[torch.arange(B, device=device), targets]
        if self.readout_type == "log_norm":
            norm = torch.linalg.vector_norm(t_states, dim=-1, keepdim=True)
            feat = torch.cat([t_states, torch.log(norm + 1e-12)], dim=-1)
        else:
            s_states = z[torch.arange(B, device=device), sources]
            cos = (s_states * t_states).sum(dim=-1, keepdim=True)
            feat = torch.cat([s_states, t_states, cos], dim=-1)
            
        return self.readout(feat)

class StructuralReadoutODE(nn.Module):
    """
    Candidate C: State-Independent Structural Readout Augmentation.
    Dynamical continuous ODE core is 100% UNMODIFIED and certified contractive (d_min >= 1.5).
    Augments the readout with non-leaking local degree properties.
    """
    def __init__(
        self,
        node_dim: int = 8,
        damping_offset: float = 1.5,
        method: str = "rk4",
        steps: int = 16,
        horizon: float = 4.0,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.damping_offset = damping_offset
        self.method = method
        self.steps = steps
        self.horizon = horizon
        
        self.w1 = nn.Linear(node_dim, node_dim, bias=False)
        self.w2 = nn.Linear(node_dim, node_dim, bias=False)
        self.log_d = nn.Parameter(torch.zeros(node_dim), requires_grad=True)
        
        # Readout consumes: [Z_t; ln(||Z_t||); log(1 + d_out(s)); log(1 + d_in(t))]
        in_dim = node_dim + 1 + 2
        self.readout = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2)
        )

    def field_parameters(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w1 = bounded_weight(self.w1.weight)
        w2 = bounded_weight(self.w2.weight)
        d = F.softplus(self.log_d) + self.damping_offset
        return w1, w2, d

    def vector_field(self, A_sparse: torch.Tensor, source: torch.Tensor, params):
        w1, w2, d = params
        N = A_sparse.shape[0]
        def field(t, z):
            B = z.shape[0]
            z_flat = z.permute(1, 0, 2).reshape(N, -1)
            prop = torch.sparse.mm(A_sparse, z_flat)
            prop = prop.reshape(N, B, self.node_dim).permute(1, 0, 2)
            act = torch.tanh(F.linear(prop, w1))
            return F.linear(act, w2) - d * z + source
        return field

    def forward(
        self,
        A_sparse: torch.Tensor,
        sources: torch.Tensor,
        targets: torch.Tensor,
        degrees: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B = sources.shape[0]
        N = A_sparse.shape[0]
        d = self.node_dim
        device = sources.device
        
        source = self.w1.weight.new_zeros((B, N, d))
        source[torch.arange(B, device=device), sources, 0] = 1.0
        
        params = self.field_parameters()
        field = self.vector_field(A_sparse, source, params)
        
        z = source.clone()
        dt = self.horizon / self.steps
        for k in range(self.steps):
            if self.method == "rk4":
                z = explicit_rk4_step(field, k * dt, z, dt)
            else:
                z = z + dt * field(k * dt, z)
                
        t_states = z[torch.arange(B, device=device), targets]
        norm = torch.linalg.vector_norm(t_states, dim=-1, keepdim=True)
        base_feat = torch.cat([t_states, torch.log(norm + 1e-12)], dim=-1)
        
        if degrees is not None:
            # degrees: (N, 2) where col 0 is out_degree, col 1 is in_degree
            s_out = degrees[sources, 0:1]
            t_in = degrees[targets, 1:2]
            deg_feat = torch.cat([torch.log1p(s_out), torch.log1p(t_in)], dim=-1)
        else:
            deg_feat = torch.zeros((B, 2), device=device)
            
        feat = torch.cat([base_feat, deg_feat], dim=-1)
        return self.readout(feat)

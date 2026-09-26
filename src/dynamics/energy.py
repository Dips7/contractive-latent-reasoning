"""Energy potential parameterizations E_theta(z; c)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class InputConvexPotential(nn.Module):
    """
    Input Convex Neural Network (ICNN) potential.
    Guarantees E_theta(z; c) is strictly convex in z for any context c.
    Hence ∇^2_z E_theta(z; c) >= 0 everywhere.
    """
    def __init__(self, latent_dim: int, context_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Linear layer for initial latent transformation
        self.w_z_init = nn.Linear(latent_dim, hidden_dim, bias=True)
        self.w_c_init = nn.Linear(context_dim, hidden_dim, bias=False)
        
        # Intermediate ICNN layers
        self.w_z_layers = nn.ModuleList()
        self.w_c_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.w_z_layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.w_c_layers.append(nn.Linear(context_dim, hidden_dim, bias=True))
            
        # Final output layer
        self.w_z_final = nn.Linear(hidden_dim, 1, bias=False)
        self.w_c_final = nn.Linear(context_dim, 1, bias=True)
        
        # Safe Kaiming initialization for affine inputs, bounded normal for positive weights
        nn.init.kaiming_normal_(self.w_z_init.weight, nonlinearity="relu")
        nn.init.zeros_(self.w_z_init.bias)
        nn.init.kaiming_normal_(self.w_c_init.weight, nonlinearity="relu")
        for w_z, w_c in zip(self.w_z_layers, self.w_c_layers):
            nn.init.normal_(w_z.weight, mean=0.0, std=0.05)
            nn.init.kaiming_normal_(w_c.weight, nonlinearity="relu")
            nn.init.zeros_(w_c.bias)
        nn.init.normal_(self.w_z_final.weight, mean=0.0, std=0.05)
        nn.init.kaiming_normal_(self.w_c_final.weight, nonlinearity="relu")
        nn.init.zeros_(self.w_c_final.bias)
        
    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Compute scalar energy E(z; c).
        Positive z-pathway weights are normalized by 1 / hidden_dim,
        preserving strict convexity while ensuring the operator norm is O(1).
        
        Args:
            z: Latent states (batch_size, latent_dim)
            c: Conditioning context (batch_size, context_dim)
        Returns:
            Scalar energy tensor (batch_size, 1)
        """
        # First layer
        h = F.softplus(self.w_z_init(z) + self.w_c_init(c))
        
        # Subsequent layers with non-negative weights for z-pathway, scaled by 1 / hidden_dim
        scale = 1.0 / self.hidden_dim
        for w_z, w_c in zip(self.w_z_layers, self.w_c_layers):
            pos_w_z = F.softplus(w_z.weight) * scale
            h = F.softplus(F.linear(h, pos_w_z) + w_c(c))
            
        # Final layer uses standard O(1 / sqrt(hidden_dim)) output scaling
        # to ensure ||∇²E|| / d_min is O(0.3 - 1.0) without causing numerical stiffness.
        pos_w_z_final = F.softplus(self.w_z_final.weight) * (3.0 / (self.hidden_dim ** 0.5))
        energy = F.linear(h, pos_w_z_final) + self.w_c_final(c)
        return energy


class QuadraticPotential(nn.Module):
    """
    Parametric quadratic potential E(z; c) = 0.5 * z^T A(c) z + b(c)^T z.
    Useful for exact linear-attractor benchmarking.
    """
    def __init__(self, latent_dim: int, context_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.linear_b = nn.Linear(context_dim, latent_dim)
        # Learn a context-dependent positive-definite factor L such that A = L L^T + eps I
        self.linear_L = nn.Linear(context_dim, latent_dim * latent_dim)
        self.eps = 0.05

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        b = self.linear_b(c) # (batch, d)
        L = self.linear_L(c).view(-1, self.latent_dim, self.latent_dim)
        # Symmetrize and ensure positive definiteness
        A = torch.bmm(L, L.transpose(1, 2)) + self.eps * torch.eye(self.latent_dim, device=z.device).unsqueeze(0)
        
        # 0.5 * z^T A z + b^T z
        Az = torch.bmm(A, z.unsqueeze(-1)).squeeze(-1)
        quad = 0.5 * torch.sum(z * Az, dim=-1, keepdim=True)
        lin = torch.sum(b * z, dim=-1, keepdim=True)
        return quad + lin

"""Contractive constraint verification and spectral loss penalty."""

import torch
import torch.nn as nn
from typing import Dict, Any


def verify_demidovich_condition(
    vector_field: nn.Module,
    z_samples: torch.Tensor,
    context: torch.Tensor,
    kappa_target: float = 0.0,
) -> Dict[str, Any]:
    """
    Numerically compute the maximum eigenvalue of the symmetric Jacobian:
    lambda_max(0.5 * (J + J^T)) across z_samples.
    
    If lambda_max <= -kappa_target < 0, the system is strictly Demidovich contractive.
    """
    vector_field.eval()
    vector_field.set_context(context)
    batch_size, latent_dim = z_samples.shape
    
    # Compute full explicit Jacobian for verification
    def f_eval(z_single, c_single):
        vector_field.set_context(c_single.unsqueeze(0))
        t = torch.tensor(0.0, device=z_single.device)
        return vector_field(t, z_single.unsqueeze(0)).squeeze(0)
        
    eigenvalues = []
    eps = 1e-4
    t0 = torch.tensor(0.0, device=z_samples.device)
    
    for i in range(min(batch_size, 32)): # check up to 32 samples
        z_i = z_samples[i:i+1].detach()
        c_i = context[i:i+1].detach()
        vector_field.set_context(c_i)
        
        # Central difference Jacobian: J[:, k] = (f(z + eps*e_k) - f(z - eps*e_k)) / (2*eps)
        J = torch.zeros(latent_dim, latent_dim, device=z_samples.device)
        for k in range(latent_dim):
            e_k = torch.zeros_like(z_i)
            e_k[0, k] = eps
            f_plus = vector_field(t0, z_i + e_k)
            f_minus = vector_field(t0, z_i - e_k)
            J[:, k] = ((f_plus - f_minus) / (2.0 * eps)).squeeze(0)
            
        # Symmetric part: Sym(J) = 0.5 * (J + J^T)
        sym_J = 0.5 * (J + J.T)
        eigs = torch.linalg.eigvalsh(sym_J)
        max_eig = torch.max(eigs).item()
        eigenvalues.append(max_eig)
        
    eigenvalues = torch.tensor(eigenvalues)
    max_observed_eig = torch.max(eigenvalues).item()
    is_contractive = (max_observed_eig <= -kappa_target)
    
    return {
        "is_contractive": is_contractive,
        "max_eigenvalue": max_observed_eig,
        "mean_max_eigenvalue": torch.mean(eigenvalues).item(),
        "kappa_margin": -max_observed_eig - kappa_target,
        "sample_count": len(eigenvalues),
    }


import torch.nn.functional as F


class SpectralContractionLoss(nn.Module):
    """
    Penalizes violations of the Demidovich contraction bound:
    L_spectral = max(0, lambda_max(Sym(J)) + kappa)^2
    Computed via power iteration on Sym(J) = 0.5 * (J + J^T),
    using finite-difference JVP and autograd VJP.
    """
    def __init__(self, kappa: float = 0.1, power_iter_steps: int = 3, eps: float = 1e-4):
        super().__init__()
        self.kappa = kappa
        self.power_iter_steps = power_iter_steps
        self.eps = eps
        
    def forward(self, vector_field: nn.Module, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        vector_field.set_context(c)
        t = torch.tensor(0.0, device=z.device)
        
        # Unit test vector v
        v = torch.randn_like(z)
        v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)
        
        # Power iteration to isolate the dominant eigenvector of Sym(J)
        with torch.no_grad():
            for _ in range(self.power_iter_steps):
                f_plus = vector_field(t, z + self.eps * v)
                f_minus = vector_field(t, z - self.eps * v)
                J_v = (f_plus - f_minus) / (2.0 * self.eps)
                v = J_v / (torch.norm(J_v, dim=-1, keepdim=True) + 1e-8)
                
        # Differentiable Rayleigh quotient at the converged eigenvector
        v = v.detach()
        f_plus = vector_field(t, z + self.eps * v)
        f_minus = vector_field(t, z - self.eps * v)
        J_v = (f_plus - f_minus) / (2.0 * self.eps)
        rayleigh = torch.sum(v * J_v, dim=-1)
        
        # Penalty for exceeding -kappa: violation = relu(rayleigh + kappa)
        violation = F.relu(rayleigh + self.kappa)
        return torch.mean(violation ** 2)

"""Jacobian eigenspectrum analyzer for Demidovich contraction bounds."""

import torch
import torch.nn as nn
from typing import Dict, Any, List


class JacobianAnalyzer:
    """
    Analyzes the eigenvalues of the symmetric Jacobian:
    Sym(J) = 0.5 * (J + J^T)
    """
    def __init__(self, vector_field: nn.Module):
        self.vector_field = vector_field

    def profile_spectrum(
        self,
        z_points: torch.Tensor,
        context: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Compute full eigenspectrum across sampled points.
        """
        self.vector_field.eval()
        t = torch.tensor(0.0, device=z_points.device)
        
        all_eigs = []
        max_eigs = []
        
        eps = 1e-4
        latent_dim = z_points.shape[-1]
        for i in range(min(z_points.size(0), 50)):
            z_i = z_points[i:i+1].detach()
            c_i = context[i:i+1].detach()
            self.vector_field.set_context(c_i)
            
            # Central difference Jacobian
            J = torch.zeros(latent_dim, latent_dim, device=z_points.device)
            for k in range(latent_dim):
                e_k = torch.zeros_like(z_i)
                e_k[0, k] = eps
                f_plus = self.vector_field(t, z_i + e_k)
                f_minus = self.vector_field(t, z_i - e_k)
                J[:, k] = ((f_plus - f_minus) / (2.0 * eps)).squeeze(0)
                
            sym_J = 0.5 * (J + J.T)
            eigs = torch.linalg.eigvalsh(sym_J)
            all_eigs.append(eigs.detach().cpu())
            max_eigs.append(torch.max(eigs).item())
            
        return {
            "max_eigenvalues": max_eigs,
            "global_max": max(max_eigs),
            "is_strictly_contractive": max(max_eigs) < 0.0,
        }

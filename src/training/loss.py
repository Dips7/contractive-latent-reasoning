"""Loss functions for contractive latent reasoning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from src.dynamics.contraction import SpectralContractionLoss


class ContractiveReasoningLoss(nn.Module):
    """
    Combined loss:
    L = L_task + lambda_eq * ||dz/dt(z*)||^2 + lambda_spec * L_spectral
    """
    def __init__(
        self,
        lambda_equilibrium: float = 0.0,
        lambda_spectral: float = 0.0,
        kappa_target: float = 0.1,
    ):
        super().__init__()
        self.lambda_equilibrium = lambda_equilibrium
        self.lambda_spectral = lambda_spectral
        self.task_loss_fn = nn.CrossEntropyLoss()
        
        if lambda_spectral > 0.0:
            self.spectral_loss_fn = SpectralContractionLoss(kappa=kappa_target)
        else:
            self.spectral_loss_fn = None

    def forward(
        self,
        model_outputs: Dict[str, Any],
        targets: torch.Tensor,
        vector_field: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        logits = model_outputs["logits"]
        z_star = model_outputs["z_star"]
        context = model_outputs["context"]
        
        # 1. Primary task loss
        if logits.dim() == 3: # autoregressive
            task_loss = self.task_loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            task_loss = self.task_loss_fn(logits, targets)
            
        # 2. Equilibrium stationary loss normalized by (1 + ||z*||^2)
        eq_loss = torch.tensor(0.0, device=logits.device)
        if self.lambda_equilibrium > 0.0 and vector_field is not None:
            t_eval = torch.tensor(0.0, device=logits.device)
            vector_field.set_context(context)
            dz_dt_terminal = vector_field(t_eval, z_star)
            # Normalized residual: ||f(z*)||^2 / (1 + ||z*||^2)
            res_sq = torch.sum(dz_dt_terminal ** 2, dim=-1)
            norm_factor = 1.0 + torch.sum(z_star ** 2, dim=-1)
            eq_loss = torch.mean(res_sq / norm_factor)
            
        # 3. Spectral penalty (if enabled)
        spec_loss = torch.tensor(0.0, device=logits.device)
        if self.lambda_spectral > 0.0 and self.spectral_loss_fn is not None and vector_field is not None:
            spec_loss = self.spectral_loss_fn(vector_field, z_star, context)
            
        total_loss = task_loss + self.lambda_equilibrium * eq_loss + self.lambda_spectral * spec_loss
        
        return {
            "total_loss": total_loss,
            "task_loss": task_loss,
            "equilibrium_loss": eq_loss,
            "spectral_loss": spec_loss,
        }

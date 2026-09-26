"""Vector field dynamics dz/dt = -∇_z E(z; c) - D·z."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DampedGradientFlowField(nn.Module):
    """
    Computes the vector field dz/dt = -∇_z E(z; c) - D·z.
    With E strictly convex (ICNN) and D strictly positive definite,
    this system is guaranteed to be strictly Demidovich contractive.
    """
    def __init__(
        self,
        energy_model: nn.Module,
        latent_dim: int,
        damping_init: float = 0.5,
        min_damping: float = 0.05,
    ):
        super().__init__()
        self.energy_model = energy_model
        self.latent_dim = latent_dim
        self.min_damping = min_damping
        
        # Learnable diagonal damping parameter log_d
        self.log_d = nn.Parameter(torch.ones(latent_dim) * torch.log(torch.tensor(damping_init)))
        self.context: Optional[torch.Tensor] = None

    def set_context(self, c: torch.Tensor):
        """Set conditioning context for ODE integration."""
        self.context = c

    def get_damping_matrix(self) -> torch.Tensor:
        """Returns diagonal damping values D_ii >= min_damping."""
        return F.softplus(self.log_d) + self.min_damping

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Evaluate dz/dt at state z and time t.
        Args:
            t: Time scalar
            z: Latent states (batch_size, latent_dim)
        Returns:
            dz/dt: (batch_size, latent_dim)
        """
        if self.context is None:
            raise ValueError("Context c must be set via set_context() before evaluating the vector field.")
            
        # Evaluate energy gradient with local grad enabled
        is_outer_grad_enabled = torch.is_grad_enabled()
        with torch.enable_grad():
            if not z.requires_grad:
                z_in = z.clone().detach().requires_grad_(True)
            else:
                z_in = z
            energy = self.energy_model(z_in, self.context) # (batch, 1)
            
            grad_outputs = torch.ones_like(energy)
            grad_z = torch.autograd.grad(
                outputs=energy,
                inputs=z_in,
                grad_outputs=grad_outputs,
                create_graph=is_outer_grad_enabled,
                retain_graph=is_outer_grad_enabled,
                only_inputs=True,
            )[0]
        
        # Linear dissipative damping D * z
        d_diag = self.get_damping_matrix()
        damping_force = d_diag * z
        
        # Damped gradient flow: dz/dt = -∇_z E - D·z
        dz_dt = -grad_z - damping_force
        return dz_dt

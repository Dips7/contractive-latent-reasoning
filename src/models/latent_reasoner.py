"""Contractive Latent Reasoner (Continuous ODE Block)."""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional

from src.dynamics.energy import InputConvexPotential
from src.dynamics.vector_field import DampedGradientFlowField
from src.dynamics.solvers import ODESolverWrapper


class ContractiveLatentReasoner(nn.Module):
    """
    Evolves latent state z(t) according to dz/dt = -∇_z E(z; c) - D·z.
    """
    def __init__(
        self,
        latent_dim: int = 128,
        context_dim: int = 128,
        energy_hidden_dim: int = 256,
        energy_layers: int = 3,
        damping_init: float = 0.5,
        min_damping: float = 0.05,
        solver_method: str = "dopri5",
        use_adjoint: bool = True,
        step_size: Optional[float] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        
        # Energy potential
        self.energy_potential = InputConvexPotential(
            latent_dim=latent_dim,
            context_dim=context_dim,
            hidden_dim=energy_hidden_dim,
            num_layers=energy_layers,
        )
        
        # Vector field
        self.vector_field = DampedGradientFlowField(
            energy_model=self.energy_potential,
            latent_dim=latent_dim,
            damping_init=damping_init,
            min_damping=min_damping,
        )
        
        # ODE solver
        self.solver = ODESolverWrapper(
            method=solver_method,
            use_adjoint=use_adjoint,
            step_size=step_size,
        )

    def forward(
        self,
        context: torch.Tensor,
        t_span: Optional[torch.Tensor] = None,
        z0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate the reasoning trajectory.
        
        Args:
            context: Conditioning vector (batch_size, context_dim)
            t_span: Integration time span, defaults to [0.0, 5.0]
            z0: Initial latent guess, defaults to zeros (batch_size, latent_dim)
            
        Returns:
            z_star: Contracted terminal state z(T) (batch_size, latent_dim)
            trajectory: Full trajectory (len(t_span), batch_size, latent_dim)
        """
        batch_size = context.shape[0]
        device = context.device
        
        if t_span is None:
            t_span = torch.tensor([0.0, 5.0], device=device)
            
        if z0 is None:
            z0 = torch.zeros(batch_size, self.latent_dim, device=device)
            
        # Bind context to vector field
        self.vector_field.set_context(context)
        
        # Integrate ODE
        trajectory = self.solver(self.vector_field, z0, t_span)
        
        # Final terminal state z*(x)
        z_star = trajectory[-1]
        return z_star, trajectory

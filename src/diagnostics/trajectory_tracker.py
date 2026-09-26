"""Trajectory tracker for verifying exponential contraction ||z1(t) - z2(t)|| <= e^(-kappa t)."""

import torch
import torch.nn as nn
from typing import Dict, Any, List


class TrajectoryTracker:
    """
    Simulates multiple trajectories from perturbed initial conditions to verify convergence.
    """
    def __init__(self, reasoner: nn.Module):
        self.reasoner = reasoner

    def evaluate_convergence(
        self,
        context: torch.Tensor,
        num_seeds: int = 10,
        t_span: torch.Tensor = None,
        noise_std: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Integrates trajectories from num_seeds different starting points z0 ~ N(0, noise_std^2 I).
        Returns pair-wise trajectory distances over time.
        """
        if t_span is None:
            t_span = torch.linspace(0.0, 10.0, 50, device=context.device)
            
        latent_dim = self.reasoner.latent_dim
        # Single context expanded across seeds
        c_expanded = context[0:1].expand(num_seeds, -1)
        z0 = torch.randn(num_seeds, latent_dim, device=context.device) * noise_std
        
        _, trajectories = self.reasoner(c_expanded, t_span=t_span, z0=z0)
        # trajectories shape: (len(t_span), num_seeds, latent_dim)
        assert trajectories.shape[0] == len(t_span), (
            f"Trajectory length {trajectories.shape[0]} does not match requested t_span length {len(t_span)}."
        )
        
        # Calculate maximum pair-wise distance at each time step
        time_steps = len(t_span)
        max_distances = []
        
        for step in range(time_steps):
            states = trajectories[step] # (num_seeds, latent_dim)
            diffs = states.unsqueeze(0) - states.unsqueeze(1) # (num_seeds, num_seeds, d)
            dists = torch.norm(diffs, dim=-1)
            max_distances.append(torch.max(dists).item())
            
        initial_distance = max_distances[0]
        final_distance = max_distances[-1]
        relative_decay = final_distance / (initial_distance + 1e-8)
        
        return {
            "t_steps": t_span.detach().cpu().numpy().tolist(),
            "max_distances": max_distances,
            "initial_max_distance": initial_distance,
            "final_max_distance": final_distance,
            "relative_decay": relative_decay,
            "converged": bool(final_distance < 1e-3 or relative_decay < 0.05),
        }

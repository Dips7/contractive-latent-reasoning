"""Tests for energy potential and vector field."""

import pytest
import torch
from src.dynamics.energy import InputConvexPotential, QuadraticPotential
from src.dynamics.vector_field import DampedGradientFlowField


def test_icnn_convexity_and_gradients():
    latent_dim = 8
    context_dim = 16
    batch_size = 4
    
    energy = InputConvexPotential(latent_dim=latent_dim, context_dim=context_dim, hidden_dim=32)
    vf = DampedGradientFlowField(energy_model=energy, latent_dim=latent_dim, damping_init=0.5)
    
    z = torch.randn(batch_size, latent_dim)
    c = torch.randn(batch_size, context_dim)
    t = torch.tensor(0.0)
    
    vf.set_context(c)
    dz_dt = vf(t, z)
    
    assert dz_dt.shape == (batch_size, latent_dim)
    assert not torch.isnan(dz_dt).any()

    # Regression test for N-2: verify initial energy gradients are strictly O(1)
    z_zero = torch.zeros(batch_size, latent_dim, requires_grad=True)
    e = energy(z_zero, c)
    grad_e = torch.autograd.grad(e.sum(), z_zero)[0]
    max_grad_norm = torch.norm(grad_e, dim=-1).max().item()
    assert max_grad_norm < 1.0, f"Initial energy gradient norm {max_grad_norm} is too large (must be O(1))"

    # Regression test: Hessian must be positive semi-definite (convex) and well-conditioned
    def f_scalar(z_in):
        return energy(z_in.unsqueeze(0), c[0:1]).squeeze()

    H = torch.autograd.functional.hessian(f_scalar, z_zero[0])
    eigs = torch.linalg.eigvalsh(H)
    assert eigs.min().item() >= -1e-5, f"Hessian has negative eigenvalue: {eigs.min().item()}"
    assert eigs.max().item() < 10.0, f"Hessian max eigenvalue {eigs.max().item()} indicates stiffness"


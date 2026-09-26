"""Tests for Demidovich contraction conditions."""

import pytest
import torch
from src.dynamics.energy import InputConvexPotential
from src.dynamics.vector_field import DampedGradientFlowField
from src.dynamics.contraction import verify_demidovich_condition


def test_demidovich_bound():
    latent_dim = 6
    context_dim = 8
    
    energy = InputConvexPotential(latent_dim=latent_dim, context_dim=context_dim, hidden_dim=16)
    # Strictly positive damping D >= 0.2
    vf = DampedGradientFlowField(energy_model=energy, latent_dim=latent_dim, damping_init=1.0, min_damping=0.2)
    
    z_samples = torch.randn(8, latent_dim)
    c_samples = torch.randn(8, context_dim)
    
    results = verify_demidovich_condition(vf, z_samples, c_samples, kappa_target=0.2)
    assert results["is_contractive"] is True
    assert results["max_eigenvalue"] <= -0.2
    assert results["sample_count"] == 8


def test_spectral_contraction_loss():
    from src.dynamics.contraction import SpectralContractionLoss

    latent_dim = 4
    context_dim = 4
    energy = InputConvexPotential(latent_dim=latent_dim, context_dim=context_dim, hidden_dim=8)
    vf = DampedGradientFlowField(energy_model=energy, latent_dim=latent_dim, damping_init=2.0, min_damping=1.0)
    
    z = torch.randn(4, latent_dim, requires_grad=True)
    c = torch.randn(4, context_dim)
    
    # Contractive field with kappa=0.0: violation should be 0.0
    loss_fn = SpectralContractionLoss(kappa=0.0)
    loss = loss_fn(vf, z, c)
    assert loss.item() == 0.0
    
    # With large kappa=10.0, rayleigh + kappa > 0, so penalty > 0 and differentiable
    loss_fn_large = SpectralContractionLoss(kappa=10.0)
    loss_large = loss_fn_large(vf, z, c)
    assert loss_large.item() > 0.0
    loss_large.backward()
    assert vf.log_d.grad is not None


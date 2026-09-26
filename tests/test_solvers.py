"""Tests for ODE solver wrapper and numerical integration."""

import pytest
import torch
from src.dynamics.solvers import ODESolverWrapper, solve_ode_rk4


def test_rk4_solver():
    # Linear ODE: dy/dt = -y -> exact solution y(t) = y0 * exp(-t)
    def linear_vf(t, y):
        return -y
        
    y0 = torch.tensor([1.0, 2.0])
    t_span = torch.tensor([0.0, 1.0])
    traj = solve_ode_rk4(linear_vf, y0, t_span, steps=100)
    
    assert traj.shape == (len(t_span), *y0.shape)
    y_final = traj[-1]
    expected = y0 * torch.exp(torch.tensor(-1.0))
    assert torch.allclose(y_final, expected, atol=1e-3)


def test_rk4_shape_contract():
    # Test arbitrary multi-step t_span shape matches (len(t_span), B, D)
    def simple_vf(t, y):
        return -0.5 * y

    B, D = 10, 16
    y0 = torch.randn(B, D)
    t_span = torch.linspace(0.0, 5.0, 30)
    
    traj = solve_ode_rk4(simple_vf, y0, t_span, steps=60)
    assert traj.shape == (30, B, D)
    
    # Check intermediate evaluation accuracy: y(t_k) = y0 * exp(-0.5 * t_k)
    for idx, t_k in enumerate(t_span):
        expected_k = y0 * torch.exp(-0.5 * t_k)
        assert torch.allclose(traj[idx], expected_k, atol=1e-3)


def test_ode_solver_wrapper_contract():
    class DummyVF(torch.nn.Module):
        def forward(self, t, y):
            return -y

    solver = ODESolverWrapper(method="rk4")
    vf = DummyVF()
    y0 = torch.randn(4, 8)
    t_span = torch.linspace(0.0, 2.0, 15)
    
    traj = solver(vf, y0, t_span)
    assert traj.shape == (len(t_span), 4, 8)


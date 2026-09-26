"""Tests for full end-to-end forward and backward passes."""

import pytest
import torch
from src.models.hybrid_model import HybridReasoningModel
from src.training.loss import ContractiveReasoningLoss


def test_hybrid_model_end_to_end():
    batch_size = 4
    seq_len = 16
    
    model = HybridReasoningModel(
        input_dim=1,
        context_dim=32,
        latent_dim=32,
        num_classes=2,
        is_autoregressive=False,
        solver_method="rk4",
    )
    
    x = torch.randn(batch_size, seq_len, 1)
    y = torch.randint(0, 2, (batch_size,))
    
    t_span = torch.tensor([0.0, 2.0])
    out = model(x, t_span=t_span)
    
    assert "logits" in out
    assert "z_star" in out
    assert out["logits"].shape == (batch_size, 2)
    assert out["z_star"].shape == (batch_size, 32)
    
    # Regression test for N-1/N-2: z_star must remain bounded O(1), not explode to 10^4
    z_star_norm = torch.norm(out["z_star"], dim=-1).mean().item()
    assert z_star_norm < 10.0, f"z_star norm {z_star_norm} exploded (must be O(1))"
    
    loss_fn = ContractiveReasoningLoss(lambda_equilibrium=0.01)
    loss_dict = loss_fn(out, y, vector_field=model.latent_reasoner.vector_field)
    
    loss = loss_dict["total_loss"]
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss_dict["task_loss"].item() < 2.0, f"Primary CE loss {loss_dict['task_loss'].item()} is too high"
    assert loss_dict["equilibrium_loss"].item() < 1.0, f"Equilibrium loss {loss_dict['equilibrium_loss'].item()} is too high"
    assert loss.item() < 5.0, f"Total loss {loss.item()} is abnormally large"
    
    # Verify backward pass computes valid gradients
    loss.backward()
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all(), "Gradients contain NaN or Inf"


def test_trainer_training_loop():
    from torch.utils.data import TensorDataset, DataLoader
    from src.training.trainer import Trainer

    model = HybridReasoningModel(
        input_dim=1,
        context_dim=16,
        latent_dim=16,
        num_classes=2,
        is_autoregressive=False,
    )
    x = torch.randn(8, 8, 1)
    y = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    loss_fn = ContractiveReasoningLoss(lambda_equilibrium=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, loss_fn, opt, device=torch.device("cpu"))

    t_span = torch.tensor([0.0, 1.0])
    stats = trainer.train_epoch(loader, t_span)
    assert 0.0 < stats["loss"] < 2.0, f"Training loss out of expected O(1) bound: {stats['loss']}"
    assert torch.isfinite(torch.tensor(stats["loss"]))
    assert 0.0 <= stats["accuracy"] <= 1.0

    eval_stats = trainer.evaluate(loader, t_span)
    assert 0.0 < eval_stats["loss"] < 2.0, f"Evaluation loss out of expected O(1) bound: {eval_stats['loss']}"
    assert torch.isfinite(torch.tensor(eval_stats["loss"]))


def test_trajectory_tracker_and_jacobian_analyzer():
    from src.models.latent_reasoner import ContractiveLatentReasoner
    from src.diagnostics.jacobian_analyzer import JacobianAnalyzer
    from src.diagnostics.trajectory_tracker import TrajectoryTracker

    latent_dim = 8
    context_dim = 16
    reasoner = ContractiveLatentReasoner(
        latent_dim=latent_dim,
        context_dim=context_dim,
        energy_hidden_dim=16,
        energy_layers=2,
        damping_init=1.0,
        min_damping=0.5,
    )

    analyzer = JacobianAnalyzer(reasoner.vector_field)
    z_pts = torch.randn(5, latent_dim)
    ctx = torch.randn(5, context_dim)
    spec_results = analyzer.profile_spectrum(z_pts, ctx)

    d_min = reasoner.vector_field.get_damping_matrix().min().item()
    assert spec_results["is_strictly_contractive"] is True
    assert spec_results["global_max"] <= -d_min + 1e-4, (
        f"global_max {spec_results['global_max']} exceeds theoretical bound {-d_min}"
    )

    tracker = TrajectoryTracker(reasoner)
    t_span = torch.linspace(0.0, 3.0, 15)
    conv_results = tracker.evaluate_convergence(ctx, num_seeds=4, t_span=t_span, noise_std=1.0)

    assert len(conv_results["max_distances"]) == 15
    assert conv_results["final_max_distance"] < 0.25 * conv_results["max_distances"][0], (
        f"Expected at least 4x contraction over t=[0, 3], got "
        f"{conv_results['max_distances'][0]:.4f} -> {conv_results['final_max_distance']:.4f}"
    )


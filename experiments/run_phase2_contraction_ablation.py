"""Executable entry point: Phase 2 Contraction Ablation and Eigenspectrum Profiler."""

import math
import sys
import time
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.models.latent_reasoner import ContractiveLatentReasoner
from src.diagnostics.jacobian_analyzer import JacobianAnalyzer
from src.diagnostics.trajectory_tracker import TrajectoryTracker
from src.utils import set_seed, save_experiment_results, load_config


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run Phase 2 Contraction Verification")
    parser.add_argument("--config", type=str, default="configs/phase2_contraction.yaml", help="Path to config file")
    parser.add_argument("--latent_dim", type=int, default=16, help="Latent state dimension")
    parser.add_argument("--context_dim", type=int, default=32, help="Context dimension")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument("--noise_std", type=float, default=None, help="Initial state perturbation standard deviation")
    parser.add_argument("--num_seeds", type=int, default=None, help="Number of trajectories for convergence test")
    args = parser.parse_args()

    # Load configuration
    cfg = {}
    if args.config and Path(args.config).exists():
        cfg = load_config(args.config)

    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)

    damping_coef = cfg.get("contraction", {}).get("damping_coefficient", 0.8)
    kappa_target = cfg.get("contraction", {}).get("kappa_target", 0.1)
    noise_std = args.noise_std if args.noise_std is not None else cfg.get("ablation", {}).get("noise_std", 2.0)
    num_seeds = args.num_seeds if args.num_seeds is not None else cfg.get("ablation", {}).get("trajectory_seeds", 5)

    device = torch.device("cpu")
    print(f"[CLR] Running Phase 2 Contraction Diagnostics on {device} (seed={seed})...")

    reasoner = ContractiveLatentReasoner(
        latent_dim=args.latent_dim,
        context_dim=args.context_dim,
        energy_hidden_dim=64,
        energy_layers=2,
        damping_init=damping_coef,
        min_damping=0.1,
    ).to(device)

    # 1. Jacobian Eigenspectrum Analysis
    print("[CLR] Step 1: Profiling Symmetric Jacobian Eigenspectrum...")
    analyzer = JacobianAnalyzer(reasoner.vector_field)
    
    context = torch.randn(10, args.context_dim)
    z_points = torch.randn(10, args.latent_dim)
    
    spec_results = analyzer.profile_spectrum(z_points, context)
    print(f"  -> Global Max Eigenvalue λ_max(Sym(J)): {spec_results['global_max']:.5f}")
    print(f"  -> Strictly Demidovich Contractive: {spec_results['is_strictly_contractive']}")

    # 2. Multi-seed Trajectory Convergence
    print(f"[CLR] Step 2: Testing Multi-Seed Trajectory Convergence (num_seeds={num_seeds}, noise_std={noise_std})...")
    tracker = TrajectoryTracker(reasoner)
    t_span = torch.linspace(0.0, 5.0, 30)
    
    conv_results = tracker.evaluate_convergence(context, num_seeds=num_seeds, t_span=t_span, noise_std=noise_std)
    theo_decay = math.exp(spec_results["global_max"] * 5.0)
    print(f"  -> Initial Max Distance (t=0.0): {conv_results['max_distances'][0]:.4f}")
    print(f"  -> Terminal Max Distance (t=5.0): {conv_results['final_max_distance']:.6f}")
    print(f"  -> Relative Distance Decay: {conv_results['relative_decay']*100:.2f}% (Theoretical bound e^(-kappa*T): {theo_decay*100:.2f}%)")
    print(f"  -> Demidovich Contraction Verified: {conv_results['relative_decay'] <= theo_decay * 1.05}")

    metrics = {
        "lambda_max_sym_j": spec_results["global_max"],
        "is_strictly_contractive": spec_results["is_strictly_contractive"],
        "initial_max_distance": conv_results["max_distances"][0],
        "terminal_max_distance": conv_results["final_max_distance"],
        "relative_decay": conv_results["relative_decay"],
        "theoretical_decay_bound": theo_decay,
        "demidovich_contraction_verified": bool(conv_results["relative_decay"] <= theo_decay * 1.05),
        "distance_decay_ratio": conv_results["max_distances"][0] / (conv_results["final_max_distance"] + 1e-12),
    }

    save_experiment_results(
        experiment_name="phase2_contraction_ablation",
        metrics=metrics,
        config={
            "latent_dim": args.latent_dim,
            "context_dim": args.context_dim,
            "damping_init": damping_coef,
            "noise_std": noise_std,
            "num_seeds": num_seeds,
            "seed": seed,
            "config_path": args.config,
        },
        seed=seed,
        start_time=start_time,
    )
    print("[CLR] Phase 2 verification complete.")


if __name__ == "__main__":
    main()

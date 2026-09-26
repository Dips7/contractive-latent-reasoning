"""Executable entry point: Phase 3 Hybrid Interface and Test-Time Compute Scaling."""

import sys
import time
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.models.hybrid_model import HybridReasoningModel
from src.utils import set_seed, save_experiment_results, load_config


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run Phase 3 Hybrid Evaluation")
    parser.add_argument("--config", type=str, default="configs/phase3_hybrid.yaml", help="Path to config file")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load configuration
    cfg = {}
    if args.config and Path(args.config).exists():
        cfg = load_config(args.config)

    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)

    print(f"[CLR] Running Phase 3 Hybrid Interface Evaluation (seed={seed})...")
    device = torch.device("cpu")

    model = HybridReasoningModel(
        input_dim=1,
        context_dim=64,
        latent_dim=64,
        num_classes=2,
        is_autoregressive=False,
    ).to(device)

    # Synthetic input sequence
    x = torch.randn(args.batch_size, 32, 1)

    print("[CLR] Evaluating Test-Time Compute Scaling (Varying Integration Horizon T):")
    scaling_results = {}
    for t_val in [1.0, 2.0, 5.0, 10.0]:
        t_span = torch.tensor([0.0, t_val])
        out = model(x, t_span=t_span)
        z_norm = torch.norm(out["z_star"], dim=-1).mean().item()
        scaling_results[f"T_{t_val:.1f}"] = {
            "z_star_norm": z_norm,
            "logits_shape": list(out["logits"].shape),
        }
        print(f"  -> T = {t_val:4.1f} | z* Mean L2 Norm: {z_norm:.4f} | Output Logits Shape: {out['logits'].shape}")

    save_experiment_results(
        experiment_name="phase3_hybrid_evaluation",
        metrics={"scaling_results": scaling_results},
        config={"batch_size": args.batch_size, "seed": seed, "config_path": args.config},
        seed=seed,
        start_time=start_time,
    )
    print("[CLR] Phase 3 evaluation complete.")


if __name__ == "__main__":
    main()

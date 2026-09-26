"""Executable entry point: Phase 1 Proof of Concept on N-bit Parity."""

import sys
import time
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import TensorDataset, DataLoader

from data.generators.parity import generate_parity_dataset
from src.models.hybrid_model import HybridReasoningModel
from src.training.loss import ContractiveReasoningLoss
from src.training.trainer import Trainer
from src.utils import set_seed, save_experiment_results, load_config


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run Phase 1 Proof-of-Concept on Parity")
    parser.add_argument("--config", type=str, default="configs/phase1_parity.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--seq_len", type=int, default=None, help="Parity sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    args = parser.parse_args()

    # Load configuration
    cfg = {}
    if args.config and Path(args.config).exists():
        cfg = load_config(args.config)

    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)

    epochs = args.epochs if args.epochs is not None else cfg.get("optimization", {}).get("epochs", 5)
    seq_len = args.seq_len if args.seq_len is not None else cfg.get("task", {}).get("seq_len", 8)
    lr = args.lr if args.lr is not None else cfg.get("optimization", {}).get("lr", 1e-3)
    latent_dim = cfg.get("model", {}).get("latent_dim", 64)
    hidden_dim = cfg.get("model", {}).get("encoder_hidden_dim", 64)

    num_train = cfg.get("task", {}).get("num_train_samples", 2000)
    num_val = cfg.get("task", {}).get("num_val_samples", 500)
    solver_method = cfg.get("dynamics", {}).get("solver", "dopri5")
    step_size = cfg.get("dynamics", {}).get("step_size", 0.05)
    kappa = cfg.get("dynamics", {}).get("contraction_bound_kappa", 0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CLR] Running Phase 1 Proof-of-Concept on device: {device} (seed={seed})")
    print(f"[CLR] Generating N={seq_len} bit parity dataset ({num_train} train, {num_val} val)...")

    x_train, y_train = generate_parity_dataset(num_samples=num_train, seq_len=seq_len, seed=seed)
    x_val, y_val = generate_parity_dataset(num_samples=num_val, seq_len=seq_len, seed=seed + 100)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=64, shuffle=False)

    print(f"[CLR] Initializing Contractive Latent Reasoning Model (latent_dim={latent_dim}, solver={solver_method})...")
    model = HybridReasoningModel(
        input_dim=1,
        context_dim=hidden_dim,
        latent_dim=latent_dim,
        num_classes=2,
        is_autoregressive=False,
        solver_method=solver_method,
        solver_step_size=step_size,
    ).to(device)

    loss_fn = ContractiveReasoningLoss(lambda_equilibrium=0.0, kappa_target=kappa)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    trainer = Trainer(model, loss_fn, optimizer, device)

    t_span = torch.tensor([0.0, 3.0], device=device)
    print(f"[CLR] Starting Training Loop for {epochs} epochs...")
    history = []
    for epoch in range(1, epochs + 1):
        train_stats = trainer.train_epoch(train_loader, t_span)
        val_stats = trainer.evaluate(val_loader, t_span)
        history.append({
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["accuracy"],
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["accuracy"],
        })
        print(f"Epoch {epoch:02d} | Train Loss: {train_stats['loss']:.4f}, Acc: {train_stats['accuracy']*100:.2f}% | "
              f"Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['accuracy']*100:.2f}%")

    metrics = {
        "final_train_loss": history[-1]["train_loss"],
        "final_train_acc": history[-1]["train_acc"],
        "final_val_loss": history[-1]["val_loss"],
        "final_val_acc": history[-1]["val_acc"],
        "history": history,
    }
    
    save_experiment_results(
        experiment_name="phase1_proof_of_concept",
        metrics=metrics,
        config={
            "epochs": epochs,
            "seq_len": seq_len,
            "lr": lr,
            "latent_dim": latent_dim,
            "seed": seed,
            "config_path": args.config,
        },
        seed=seed,
        start_time=start_time,
    )
    print("[CLR] Phase 1 run completed successfully.")


if __name__ == "__main__":
    main()

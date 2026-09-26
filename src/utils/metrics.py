"""Evaluation metrics."""

import torch
from typing import Dict, Any


def compute_classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Computes accuracy, cross-entropy loss, and top-1 predictions.
    """
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    
    return {
        "accuracy": correct / max(1, total),
        "total_samples": total,
    }

"""Utility to persist experimental results, metadata, and environment fingerprints."""

import os
import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import torch


import math


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively converts NaN and Infinity to None for strict RFC 8259 compliance."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def save_experiment_results(
    experiment_name: str,
    metrics: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    start_time: Optional[float] = None,
    output_dir: Optional[str] = None,
) -> Path:
    """
    Saves a comprehensive JSON run record for scientific reproducibility.
    Includes hardware info, package versions, seed, metrics, and wall time.
    Strictly RFC 8259 compliant (no bare Infinity/NaN tokens).
    """
    elapsed = time.time() - start_time if start_time is not None else None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    if output_dir is None:
        results_dir = Path(__file__).resolve().parents[2] / "experiments" / "results"
    else:
        results_dir = Path(output_dir)
        
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Check torchdiffeq version
    torchdiffeq_ver = None
    try:
        import torchdiffeq
        torchdiffeq_ver = getattr(torchdiffeq, "__version__", "installed")
    except ImportError:
        torchdiffeq_ver = "not_installed"
        
    payload = {
        "experiment": experiment_name,
        "timestamp_utc": timestamp,
        "seed": seed,
        "wall_time_seconds": elapsed,
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "torchdiffeq_version": torchdiffeq_ver,
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        },
        "config": config or {},
        "metrics": metrics,
    }
    
    sanitized_payload = _sanitize_for_json(payload)
    
    filename = f"{experiment_name}_{timestamp}.json"
    file_path = results_dir / filename
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sanitized_payload, f, indent=2, allow_nan=False)
        
    print(f"[Results Persisted] Saved execution artifact to {file_path}")
    return file_path

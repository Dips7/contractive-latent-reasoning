"""Configuration loading and merging utility for contractive reasoning experiments."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges override into base dictionary."""
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(config_path: str, base_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads YAML config and resolves defaults inheritance (e.g. `defaults: - base`).
    """
    cfg_file = Path(config_path)
    if not cfg_file.is_absolute() and base_dir is not None:
        cfg_file = Path(base_dir) / cfg_file
    elif not cfg_file.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        cfg_file = repo_root / config_path

    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Check for defaults
    defaults = cfg.get("defaults", [])
    if isinstance(defaults, list):
        merged_base: Dict[str, Any] = {}
        config_dir = cfg_file.parent
        for default_name in defaults:
            if isinstance(default_name, str):
                default_path = config_dir / f"{default_name}.yaml"
                if default_path.exists():
                    parent_cfg = load_config(str(default_path), base_dir=str(config_dir))
                    merged_base = _deep_merge(merged_base, parent_cfg)
        cfg_without_defaults = {k: v for k, v in cfg.items() if k != "defaults"}
        cfg = _deep_merge(merged_base, cfg_without_defaults)

    return cfg

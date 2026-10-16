"""Reproducibility helpers: seeding and environment recording."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from typing import Any, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def record_environment(extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Record versions, hardware, git and model-loading-relevant facts."""
    import importlib.metadata as md

    def _ver(pkg: str) -> Optional[str]:
        try:
            return md.version(pkg)
        except Exception:
            return None

    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_mb": (
            torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            if torch.cuda.is_available()
            else None
        ),
        "cpu_count": os.cpu_count(),
        "transformer_lens": _ver("transformer-lens"),
        "transformers": _ver("transformers"),
        "accelerate": _ver("accelerate"),
        "einops": _ver("einops"),
        "numpy": _ver("numpy"),
        "git_commit": _git_commit(),
        "working_directory": os.getcwd(),
    }
    if extra:
        env["extra"] = extra
    return env


def save_environment(path: str, extra: Optional[dict[str, Any]] = None) -> None:
    with open(path, "w") as f:
        json.dump(record_environment(extra), f, indent=2, ensure_ascii=False)

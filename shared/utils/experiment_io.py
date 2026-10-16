"""Run-directory management and JSON/YAML IO.

Every Q1 run writes to questions/q1_private_communication/results/<run_id>/
with a fixed layout (resolved_config.yaml, environment.json, checkpoints/,
logs/, predictions/, statistics/, figures/, report.md). Existing runs are
never overwritten: a new run gets a fresh run_id.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import yaml


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_yaml(path: str | Path) -> Any:
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


class RunDirectory:
    """Creates and namespaces a single experiment run directory.

    run_id format: <tag>-<YYYYmmdd-HHMMSS>[-<n>]; never reuses an existing id.
    """

    def __init__(self, base_dir: str | Path, tag: str = "run") -> None:
        self.base = Path(base_dir)
        self.tag = tag
        self.run_id = self._fresh_id()
        self.path = self.base / self.run_id

    def _fresh_id(self) -> str:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = f"{self.tag}-{stamp}"
        cand = base
        i = 1
        while (self.base / cand).exists():
            i += 1
            cand = f"{base}-{i}"
        return cand

    def make(self) -> Path:
        for sub in [
            "checkpoints", "logs", "predictions", "statistics",
            "figures",
        ]:
            (self.path / sub).mkdir(parents=True, exist_ok=True)
        return self.path

    def sub(self, name: str) -> Path:
        return self.path / name

    def resolved_config_path(self) -> Path:
        return self.path / "resolved_config.yaml"

    def save_resolved_config(self, cfg: dict[str, Any]) -> None:
        save_yaml(cfg, self.resolved_config_path())


def ensure_abspath(path: str | Path, root: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(root) / p


def default_env_path(name: str, default: str) -> str:
    """Config strings may reference env vars; this resolves them."""
    return os.environ.get(name, default)

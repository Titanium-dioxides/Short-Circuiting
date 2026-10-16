"""Config resolution: base YAML + optional variant overlay -> resolved
config. The resolved config must contain all final effective parameters and
keeps architecture_mode / intervention_mode / budget_regime as SEPARATE
fields (spec 15.8)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from shared.models import validate_budget_ranks
from shared.utils.experiment_io import load_yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(name: str, variant: Optional[str] = None) -> dict:
    base = load_yaml(CONFIG_DIR / name)
    if variant:
        base = deep_merge(base, load_yaml(CONFIG_DIR / "variants" / variant))
    return base


def resolve_config(cfg: dict, circuit, d_model: int = 768, d_head: int = 64) -> dict:
    """Attach computed fields: budget tables per regime, ranks, param
    counts, and the separated mode fields."""
    out = deep_merge(dict(cfg), {})
    in_dim = circuit.n_a * d_model
    out["circuit"] = circuit.to_dict()
    out["computed"] = {}
    for regime in cfg["budget_regimes"]:
        budget_cfg = cfg["budget"][regime]
        from shared.models import BudgetConfig

        b = BudgetConfig(regime=regime, **budget_cfg)
        table = validate_budget_ranks(b, in_dim, d_head, d_model, circuit.n_b)
        out["computed"][regime] = {
            "budget_config": budget_cfg,
            "ranks_and_param_counts": table,
            "in_dim": in_dim,
            "d_head": d_head,
            "d_model": d_model,
            "n_a": circuit.n_a,
            "n_b": circuit.n_b,
        }
    out["field_separation"] = {
        "architecture_mode": "per-group field, one of: original/delete/private_q/private_v/private_qv/public_adapter/constant_q/constant_v/constant_qv",
        "intervention_mode": "per-causal-test field, one of: normal/off/mean_input/shuffled_input/paired_swap/q_on_v_off/q_off_v_on/swap_q_only/swap_v_only/swap_both",
        "budget_regime": cfg["budget_regimes"],
        "private_mode": cfg["private_mode"],
    }
    out["tuning_budget"] = (
        "none: no hyperparameter search was run; learning rate/steps/batch "
        "are fixed in the config for ALL groups (identical supervision, step "
        "cap and early-stopping criterion per spec 8/9)"
    )
    return out


def data_fingerprint(cfg: dict, templates: tuple, holdout: tuple, pools: dict) -> str:
    """Deterministic id for a generated dataset, shared across runs."""
    payload = {
        "data": cfg["data"],
        "templates": list(templates),
        "holdout": list(holdout),
        "pools": pools,
    }
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:12]

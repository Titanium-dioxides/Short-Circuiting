"""Statistics: paired bootstrap and the recovery-rate definition.

Recovery rate (spec 11):
    R = (mean D_private - mean D_delete) / (mean D_original - mean D_delete)
computed as a ratio of means (never averaged per-sample ratios).

Denominator validity rule (frozen BEFORE final testing, 2026-10-16):
the denominator (mean D_original - mean D_delete) must be > 0 and
>= DENOMINATOR_MIN, otherwise R is reported as "not interpretable".

Bootstrap: resampling unit = pair_id (the jointly-generated base sample),
never individual samples, never stacked evaluations of the same test sample
across seeds.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

DENOMINATOR_MIN = 0.05  # logit scale; frozen before final testing


def recovery_rate(
    mean_d_original: float,
    mean_d_delete: float,
    mean_d_private: float,
) -> tuple[Optional[float], bool, str]:
    """Returns (R, interpretable, reason)."""
    denom = mean_d_original - mean_d_delete
    if not (denom > 0):
        return None, False, f"denominator {denom:.4f} <= 0: delete caused no damage"
    if denom < DENOMINATOR_MIN:
        return None, False, (
            f"denominator {denom:.4f} < {DENOMINATOR_MIN}: deletion damage too "
            "small for an interpretable recovery rate"
        )
    return (mean_d_private - mean_d_delete) / denom, True, "ok"


def bootstrap_paired_ci(
    pair_ids: Sequence,
    statistic_fn: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Paired bootstrap over pair_id units.

    `pair_ids`: per-sample pair ids (same length as the statistic's rows).
    `statistic_fn(pair_ids_sel) -> float`: computes the statistic from the
    samples whose pair ids are in the selection.
    """
    rng = np.random.default_rng(seed)
    ids = np.asarray(pair_ids)
    est = float(statistic_fn(ids))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sel = rng.choice(ids, size=ids.shape[0], replace=True)
        boot[i] = statistic_fn(sel)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return {"estimate": est, "ci_low": lo, "ci_high": hi, "n_boot": n_boot}

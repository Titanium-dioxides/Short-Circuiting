"""Q1 metrics (spec 11/19): D, two accuracies, NLL, recovery rate,
per-sample records and grouped statistics.

D_i = logit(correct_i) - logit(distractor_i) at the final input position.
Recovery rate: R = (meanD_priv - meanD_delete) / (meanD_orig - meanD_delete),
a ratio of means. The denominator validity rule lives in
shared/evaluation/statistics.py and is frozen before final testing.
"""

from __future__ import annotations

from typing import Optional

import torch

from shared.evaluation.statistics import (
    bootstrap_paired_ci,
    recovery_rate,
)


def per_sample_metrics(
    logits: torch.Tensor, samples: list, final_pos: int = -1
) -> list[dict]:
    """logits [B, S, V]; samples: list of IOSample (or dicts)."""
    def g(obj, key):
        return getattr(obj, key) if hasattr(obj, key) else obj[key]

    out = []
    with torch.no_grad():
        logp = torch.log_softmax(logits[:, final_pos, :], dim=-1)
        top1 = logits[:, final_pos, :].argmax(dim=-1)
    for i, s in enumerate(samples):
        correct = g(s, "correct_token")
        distractor = g(s, "distractor_token")
        d = logits[i, final_pos, correct] - logits[i, final_pos, distractor]
        rec = {
            "sample_id": g(s, "sample_id"),
            "pair_id": g(s, "pair_id"),
            "split": g(s, "split"),
            "template_id": g(s, "template_id"),
            "order": g(s, "order"),
            "form": g(s, "form"),
            "correct_token": int(correct),
            "distractor_token": int(distractor),
            "D": float(d.item()),
            "two_way_correct": bool(d.item() > 0),
            "top1_correct": bool(top1[i].item() == correct),
            "NLL": float(-logp[i, correct].item()),
            "top1_token": int(top1[i].item()),
        }
        out.append(rec)
    return out


def aggregate(records: list[dict]) -> dict:
    import numpy as np

    d = np.array([r["D"] for r in records])
    nll = np.array([r["NLL"] for r in records])
    return {
        "n": len(records),
        "mean_D": float(d.mean()),
        "two_way_acc": float(np.mean([r["two_way_correct"] for r in records])),
        "top1_acc": float(np.mean([r["top1_correct"] for r in records])),
        "mean_NLL": float(nll.mean()),
    }


def aggregate_by(records: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(str(r[key]), []).append(r)
    return {k: aggregate(v) for k, v in sorted(groups.items())}


def group_stats(records: list[dict]) -> dict:
    return {
        "overall": aggregate(records),
        "by_template": aggregate_by(records, "template_id"),
        "by_order": aggregate_by(records, "order"),
        "by_form": aggregate_by(records, "form"),
    }


def recovery_metrics(
    records_original: list[dict],
    records_delete: list[dict],
    records_private: list[dict],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Recovery rate + paired bootstrap CIs (resampling unit = pair_id)."""
    ids = [r["pair_id"] for r in records_original]
    by_pair_orig = {r["pair_id"]: r for r in records_original}
    by_pair_del = {r["pair_id"]: r for r in records_delete}
    by_pair_priv = {r["pair_id"]: r for r in records_private}

    def mean_d(sel_ids):
        return float(
            sum(by_pair_orig[i]["D"] for i in sel_ids) / len(sel_ids)
        )

    ci_orig = bootstrap_paired_ci(ids, mean_d, n_boot=n_boot, seed=seed)

    def mean_d_del(sel_ids):
        return float(sum(by_pair_del[i]["D"] for i in sel_ids) / len(sel_ids))

    ci_del = bootstrap_paired_ci(ids, mean_d_del, n_boot=n_boot, seed=seed)

    def mean_d_priv(sel_ids):
        return float(sum(by_pair_priv[i]["D"] for i in sel_ids) / len(sel_ids))

    ci_priv = bootstrap_paired_ci(ids, mean_d_priv, n_boot=n_boot, seed=seed)

    m_orig, m_del, m_priv = ci_orig["estimate"], ci_del["estimate"], ci_priv["estimate"]
    R, interpretable, reason = recovery_rate(m_orig, m_del, m_priv)

    # bootstrap the recovery rate itself (ratio of means over resampled pairs)
    import numpy as np

    rng = np.random.default_rng(seed)
    id_arr = np.asarray(ids)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sel = rng.choice(id_arr, size=id_arr.shape[0], replace=True)
        r, ok, _ = recovery_rate(
            mean_d(sel), mean_d_del(sel), mean_d_priv(sel)
        )
        boots[i] = r if (ok and r is not None) else np.nan
    finite = boots[np.isfinite(boots)]
    r_ci = {
        "estimate": R,
        "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else None,
        "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else None,
        "n_boot": n_boot,
        "interpretable": interpretable,
        "reason": reason,
        "denominator": m_orig - m_del,
    }
    return {
        "mean_D": {"original": m_orig, "delete": m_del, "private": m_priv},
        "mean_D_ci": {
            "original": {k: ci_orig[k] for k in ("estimate", "ci_low", "ci_high")},
            "delete": {k: ci_del[k] for k in ("estimate", "ci_low", "ci_high")},
            "private": {k: ci_priv[k] for k in ("estimate", "ci_low", "ci_high")},
        },
        "recovery_rate": r_ci,
    }

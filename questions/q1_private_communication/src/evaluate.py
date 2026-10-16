"""Q1 evaluation (spec 11/19): generalization evaluation for every
architecture mode x seed on all test splits, per-sample prediction files,
grouped statistics, recovery rates vs original/delete, and gain vs the
static-compensation controls (constant_q/v/qv)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

from shared.models import BudgetRanks, CircuitSpec, PrivateTransformer
from shared.training.checkpoint import load_private_checkpoint
from shared.utils.experiment_io import save_json
from questions.q1_private_communication.src.data import (
    IOIDataset,
    make_bucketed_loader,
)
from questions.q1_private_communication.src.metrics import (
    aggregate,
    group_stats,
    per_sample_metrics,
    recovery_metrics,
)

NO_TRAIN_MODES = ("original", "delete")


def build_trained_model(backbone, circuit, mode: str, checkpoint: Optional[str]) -> PrivateTransformer:
    if mode in NO_TRAIN_MODES or checkpoint is None:
        return PrivateTransformer(backbone, circuit, mode, BudgetRanks())
    meta = torch.load(Path(checkpoint).with_suffix(".meta.pt"),
                      map_location="cpu", weights_only=False)
    ranks = BudgetRanks(
        q_rank=meta["ranks"].get("q_rank"),
        v_rank=meta["ranks"].get("v_rank"),
        pub_rank=meta["ranks"].get("pub_rank"),
    )
    model = PrivateTransformer(backbone, circuit, mode, ranks, seed=meta["meta"].get("seed", 0))
    load_private_checkpoint(model, checkpoint)
    model.eval()
    return model


def evaluate_group(
    backbone,
    circuit: CircuitSpec,
    mode: str,
    seed: int,
    dataset: IOIDataset,
    splits: list[str],
    checkpoint: Optional[str],
    run_dir: Path,
    device: str = "cuda",
    batch_size: int = 32,
) -> dict:
    model = build_trained_model(backbone, circuit, mode, checkpoint)
    model.eval()
    result: dict = {"mode": mode, "seed": seed, "splits": {}}
    with torch.no_grad():
        for split in splits:
            recs = []
            loader = make_bucketed_loader(dataset, split, batch_size)
            for chunk, toks in loader():
                logits, _ = model.forward(toks)
                for r in per_sample_metrics(logits, chunk):
                    r["group"] = mode
                    r["mode"] = mode
                    r["seed"] = seed
                    r["intervention"] = "normal"
                    r["donor_id"] = None
                    recs.append(r)
            result["splits"][split] = {
                "overall": aggregate(recs),
                "groups": group_stats(recs),
            }
            save_json(
                recs,
                run_dir / "predictions" / f"{mode}-seed{seed}-{split}.json",
            )
    return result


def run_evaluation_suite(
    backbone,
    circuit: CircuitSpec,
    dataset: IOIDataset,
    modes: list[str],
    seeds: tuple[int, ...],
    training_summary: dict,
    splits: list[str],
    run_dir: Path,
    device: str = "cuda",
    batch_size: int = 32,
) -> dict:
    """Evaluate all groups and compute recovery rates per split."""
    report: dict = {"groups": [], "recovery": {}}
    ckpt_by_mode_seed = {}
    for g in training_summary.get("groups", []):
        if not g.get("not_executed"):
            ckpt_by_mode_seed[(g["mode"], g["seed"])] = g.get("best_checkpoint")

    evals: dict[tuple[str, int], dict] = {}
    for mode in modes:
        seeds_for_mode = seeds if mode not in NO_TRAIN_MODES else (0,)
        for seed in seeds_for_mode:
            ckpt = ckpt_by_mode_seed.get((mode, seed))
            if mode not in NO_TRAIN_MODES and ckpt is None:
                entry = {"mode": mode, "seed": seed, "not_executed": True,
                         "note": "no checkpoint found"}
                report["groups"].append(entry)
                continue
            res = evaluate_group(backbone, circuit, mode, seed, dataset, splits,
                                 ckpt, run_dir, device, batch_size)
            evals[(mode, seed)] = res
            report["groups"].append(res)

    # recovery rates per split, per seed, per private mode (spec 11)
    for split in splits:
        report["recovery"][split] = {}
        for seed in seeds:
            o = evals.get(("original", 0))
            d = evals.get(("delete", 0))
            if o is None or d is None:
                continue
            recs_o = _load_recs(run_dir, "original", 0, split)
            recs_d = _load_recs(run_dir, "delete", 0, split)
            report["recovery"][split][f"seed{seed}"] = {}
            for mode in ("private_q", "private_v", "private_qv"):
                e = evals.get((mode, seed))
                if e is None or "not_executed" in e:
                    report["recovery"][split][f"seed{seed}"][mode] = {
                        "not_executed": True
                    }
                    continue
                recs_p = _load_recs(run_dir, mode, seed, split)
                report["recovery"][split][f"seed{seed}"][mode] = recovery_metrics(
                    recs_o, recs_d, recs_p
                )
            # gain vs static controls (spec 19)
            report["recovery"][split][f"seed{seed}"]["gain_vs_static"] = {}
            for mode in ("private_q", "private_v", "private_qv"):
                for static in ("constant_q", "constant_v", "constant_qv"):
                    e_s = evals.get((static, seed))
                    if e_s is None:
                        continue
                    recs_priv = _load_recs(run_dir, mode, seed, split)
                    recs_static = _load_recs(run_dir, static, seed, split)
                    gain = (
                        aggregate(recs_priv)["mean_D"]
                        - aggregate(recs_static)["mean_D"]
                    )
                    report["recovery"][split][f"seed{seed}"]["gain_vs_static"][
                        f"{mode}_vs_{static}"
                    ] = gain
    save_json(report, run_dir / "statistics" / "evaluation.json")
    return report


def _load_recs(run_dir: Path, mode: str, seed: int, split: str) -> list[dict]:
    path = run_dir / "predictions" / f"{mode}-seed{seed}-{split}.json"
    with open(path) as f:
        return json.load(f)

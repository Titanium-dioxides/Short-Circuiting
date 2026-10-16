"""Q1 training (spec 9).

Loss: L_task = -log_softmax(logits)[:, final_pos, correct] (mean).
Optional: L = L_task + lambda_KL * KL(p_teacher || p_student), lambda_KL=0 by
default; the teacher (original model) is supervision only, never a private
input. Only adapter/constant parameters are updated; the backbone stays
frozen and is fingerprint-verified. Hyperparameters and checkpoints are
selected on validation only. All groups share data, supervision, step cap,
early-stopping criterion and tuning budget.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from shared.models import (
    BudgetConfig,
    BudgetRanks,
    CircuitSpec,
    PrivateTransformer,
    resolve_budget,
)
from shared.models.backbone import param_fingerprint, fingerprint_matches
from shared.training.checkpoint import save_private_checkpoint
from shared.utils.experiment_io import save_json
from shared.utils.reproducibility import set_seed
from questions.q1_private_communication.src.data import (
    IOIDataset,
    make_bucketed_loader,
)
from questions.q1_private_communication.src.metrics import (
    aggregate,
    per_sample_metrics,
)


@dataclass
class TrainingConfig:
    architecture_mode: str
    budget: BudgetConfig
    lr: float = 1e-3
    steps: int = 2000
    batch_size: int = 32
    eval_every: int = 100
    early_stop_patience: int = 10
    lambda_kl: float = 0.0
    kl_temperature: float = 1.0
    seed: int = 0
    device: str = "cuda"
    grad_clip: float = 1.0
    max_eval_batches: Optional[int] = None
    selection_metric: str = "mean_D"  # checkpoint selection on validation D


@dataclass
class TrainingResult:
    mode: str
    seed: int
    best_val_metric: Optional[float] = None
    steps_done: int = 0
    wall_clock_s: float = 0.0
    peak_gpu_mb: Optional[float] = None
    param_stats: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    best_checkpoint: Optional[str] = None
    early_stopped: bool = False
    not_executed: bool = False
    note: str = ""


def build_ranks(cfg: TrainingConfig, circuit: CircuitSpec, d_head: int, d_model: int) -> BudgetRanks:
    in_dim = circuit.n_a * d_model
    if cfg.architecture_mode == "public_adapter":
        # public_adapter is sized to the corresponding private model's actual
        # budget (spec 16): the private_qv total under the same budget config
        from shared.models.routing import param_counts_for_mode

        qv_ranks = resolve_budget(
            cfg.budget, "private_qv", in_dim, d_head, d_model, circuit.n_b
        )
        ref_total = param_counts_for_mode(
            "private_qv", qv_ranks, in_dim, d_head, d_model, circuit.n_b
        )["total"]
        return resolve_budget(
            cfg.budget, "public_adapter", in_dim, d_head, d_model, circuit.n_b,
            ref_total=ref_total,
        )
    return resolve_budget(cfg.budget, cfg.architecture_mode, in_dim, d_head, d_model, circuit.n_b)


def train_group(
    backbone,
    circuit: CircuitSpec,
    cfg: TrainingConfig,
    dataset: IOIDataset,
    run_dir: Path,
) -> TrainingResult:
    set_seed(cfg.seed)
    t0 = time.time()
    ranks = build_ranks(cfg, circuit, backbone.cfg.d_head, backbone.cfg.d_model)
    model = PrivateTransformer(backbone, circuit, cfg.architecture_mode, ranks, seed=cfg.seed)
    model.train()
    # spec 5: the model stays in eval mode (dropout off); only the adapter
    # parameters are trained (they contain no dropout/BatchNorm anyway).
    model.backbone.eval()

    fp0 = param_fingerprint(backbone)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=cfg.lr)
    result = TrainingResult(mode=cfg.architecture_mode, seed=cfg.seed)
    result.param_stats = model.param_stats()

    best_metric: Optional[float] = None
    best_step = 0
    patience = 0
    step = 0
    global_step = 0
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{cfg.architecture_mode}-seed{cfg.seed}.pt"

    def evaluate(split: str, max_batches: Optional[int]):
        model.eval()
        recs = []
        nll_total = 0.0
        nll_n = 0
        loader = make_bucketed_loader(dataset, split, cfg.batch_size)
        with torch.no_grad():
            for chunk, toks in loader():
                logits, _ = model.forward(toks)
                recs.extend(per_sample_metrics(logits, chunk))
                lp = F.log_softmax(logits[:, -1, :], dim=-1)
                nll_total += -lp[
                    torch.arange(logits.shape[0]),
                    torch.tensor([s.correct_token for s in chunk], device=logits.device),
                ].sum().item()
                nll_n += len(chunk)
                if max_batches and len(recs) >= max_batches * cfg.batch_size:
                    recs = recs[: max_batches * cfg.batch_size]
                    break
        model.train()
        model.backbone.eval()  # keep dropout off (spec 5)
        agg = aggregate(recs)
        agg["NLL_mean_over_batches"] = nll_total / max(nll_n, 1)
        return agg

    # pre-training integrity: zero-init model == delete (spec 9)
    with torch.no_grad():
        delete_model = PrivateTransformer(backbone, circuit, "delete", BudgetRanks())
        toks_probe = dataset.tensor_batch(
            [dataset.by_split("validation")[0]]
        ).to(cfg.device)
        l0, _ = model.forward(toks_probe)
        l1, _ = delete_model.forward(toks_probe)
    if not torch.allclose(l0, l1, atol=1e-6, rtol=1e-6):
        raise RuntimeError(
            f"{cfg.architecture_mode}: zero-init != delete before training"
        )

    train_loader = make_bucketed_loader(
        dataset, "train", cfg.batch_size, shuffle=True, seed=cfg.seed
    )
    while step < cfg.steps:
        epoch_loss = 0.0
        epoch_n = 0
        for chunk, toks in train_loader():
            if step >= cfg.steps:
                break
            toks = toks.to(cfg.device)
            correct = torch.tensor([s.correct_token for s in chunk], device=toks.device)
            logits, _ = model.forward(toks)
            logp = F.log_softmax(logits[:, -1, :], dim=-1)
            loss = -logp[torch.arange(logits.shape[0]), correct].mean()
            if cfg.lambda_kl > 0:
                with torch.no_grad():
                    t_logits = backbone(toks)[:, -1, :]
                kl = F.kl_div(
                    F.log_softmax(logits[:, -1, :] / cfg.kl_temperature, dim=-1),
                    F.softmax(t_logits / cfg.kl_temperature, dim=-1),
                    reduction="batchmean",
                )
                loss = loss + cfg.lambda_kl * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # record gradient presence (spec 9: valid gradients)
            grad_norms = {
                n: p.grad.norm().item()
                for n, p in model.adapters.named_parameters()
                if p.grad is not None
            } if model.adapters is not None else {}
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.grad_clip)
            opt.step()
            epoch_loss += float(loss.item()) * len(chunk)
            epoch_n += len(chunk)
            global_step += 1
            step += 1

            if step % cfg.eval_every == 0 or step == cfg.steps:
                val = evaluate("validation", cfg.max_eval_batches)
                metric = val[cfg.selection_metric]
                result.history.append({
                    "step": step,
                    "train_loss": epoch_loss / max(epoch_n, 1),
                    "val": val,
                    "grad_norm_min": (
                        min(grad_norms.values()) if grad_norms else None
                    ),
                })
                if best_metric is None or metric > best_metric:
                    best_metric = metric
                    best_step = step
                    patience = 0
                    save_private_checkpoint(model, ckpt_path, meta={
                        "step": step, "val": val, "seed": cfg.seed,
                    })
                else:
                    patience += 1
                if patience >= cfg.early_stop_patience:
                    result.early_stopped = True
                    break

        # end-of-epoch integrity: backbone untouched (spec 9)
        if not fingerprint_matches(param_fingerprint(backbone), fp0):
            raise RuntimeError("backbone parameters changed during training!")
        if result.early_stopped:
            break

    result.steps_done = step
    result.wall_clock_s = time.time() - t0
    result.best_val_metric = best_metric
    result.best_checkpoint = str(ckpt_path) if best_metric is not None else None
    if torch.cuda.is_available() and cfg.device == "cuda":
        result.peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return result


def run_training_suite(
    backbone,
    circuit: CircuitSpec,
    dataset: IOIDataset,
    modes: list[str],
    budget: BudgetConfig,
    base_cfg: TrainingConfig,
    run_dir: Path,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> dict:
    """Train every architecture mode x seed (spec 14 order step 9)."""
    report = {"groups": [], "budget": budget.regime}
    for mode in modes:
        if mode in ("original", "delete"):
            # no trainable parameters; evaluated directly at eval time
            report["groups"].append({
                "mode": mode, "seed": 0,
                "not_trained": "no trainable parameters",
                "not_executed": False, "note": "",
            })
            continue
        for seed in seeds:
            cfg = TrainingConfig(
                architecture_mode=mode,
                budget=budget,
                lr=base_cfg.lr,
                steps=base_cfg.steps,
                batch_size=base_cfg.batch_size,
                eval_every=base_cfg.eval_every,
                early_stop_patience=base_cfg.early_stop_patience,
                lambda_kl=base_cfg.lambda_kl,
                kl_temperature=base_cfg.kl_temperature,
                seed=seed,
                device=base_cfg.device,
                grad_clip=base_cfg.grad_clip,
                max_eval_batches=base_cfg.max_eval_batches,
                selection_metric=base_cfg.selection_metric,
            )
            res = train_group(backbone, circuit, cfg, dataset, run_dir)
            entry = {
                "mode": res.mode,
                "seed": res.seed,
                "best_val_metric": res.best_val_metric,
                "steps_done": res.steps_done,
                "wall_clock_s": res.wall_clock_s,
                "peak_gpu_mb": res.peak_gpu_mb,
                "param_stats": res.param_stats,
                "best_checkpoint": res.best_checkpoint,
                "early_stopped": res.early_stopped,
                "not_executed": res.not_executed,
                "note": res.note,
            }
            report["groups"].append(entry)
    save_json(report, run_dir / "statistics" / "training_summary.json")
    return report

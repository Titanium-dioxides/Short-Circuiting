"""Q1 CLI (spec 13): directly copy-pasteable commands.

  python -m questions.q1_private_communication.src.cli env
  python -m questions.q1_private_communication.src.cli test
  python -m questions.q1_private_communication.src.cli data --config smoke.yaml
  python -m questions.q1_private_communication.src.cli baseline --config smoke.yaml
  python -m questions.q1_private_communication.src.cli train --config smoke.yaml
  python -m questions.q1_private_communication.src.cli evaluate --config smoke.yaml
  python -m questions.q1_private_communication.src.cli causal --config smoke.yaml
  python -m questions.q1_private_communication.src.cli report --config smoke.yaml
  python -m questions.q1_private_communication.src.cli smoke --config smoke.yaml

State files (data, circuit) live under questions/q1_private_communication/data/;
run artifacts under results/<run_id>/ and never overwrite existing runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from shared.models import CircuitSpec, HeadRef, load_backbone, load_tokenizer
from shared.utils.experiment_io import RunDirectory, load_json, save_json, save_yaml
from shared.utils.reproducibility import record_environment, set_seed
from questions.q1_private_communication.src.configs import (
    data_fingerprint,
    load_config,
    resolve_config,
)
from questions.q1_private_communication.src.data import (
    NAME_POOLS,
    TEMPLATES_HOLDOUT,
    TEMPLATES_TRAIN,
    IOIDataset,
    generate_ioi,
)

Q1_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Q1_ROOT / "data"
RESULTS_DIR = Q1_ROOT / "results"


def _data_path(cfg: dict) -> Path:
    fp = data_fingerprint(cfg, TEMPLATES_TRAIN, TEMPLATES_HOLDOUT, NAME_POOLS)
    return DATA_DIR / f"ioi-{fp}.json"


def _circuit_path(cfg: dict) -> Path:
    return _data_path(cfg).with_name(
        _data_path(cfg).stem + "-selected_circuit.json"
    )


def _load_circuit(path: Path) -> CircuitSpec:
    sel = load_json(path)
    a = [HeadRef(int(x.split("H")[0][1:]), int(x.split("H")[1])) for x in sel["selected_a"]]
    b = [HeadRef(int(x.split("H")[0][1:]), int(x.split("H")[1])) for x in sel["selected_b"]]
    return CircuitSpec(a_heads=tuple(a), b_heads=tuple(b),
                       source=sel["selected_circuit"]["source"],
                       selection_notes=sel["selected_circuit"]["selection_notes"])


def _load_backbone(cfg: dict, device: str):
    return load_backbone(device=device)


def cmd_env(args):
    save_json(record_environment(), Q1_ROOT / "data" / "environment.json")
    print(json.dumps(record_environment(), indent=2))


def cmd_test(args):
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/shared",
         "questions/q1_private_communication/tests", "-q"],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
    )


def cmd_data(args):
    cfg = load_config(args.config, args.variant)
    set_seed(cfg["data"]["seed"])
    tok = load_tokenizer()
    ds = generate_ioi(
        tok,
        seed=cfg["data"]["seed"],
        n_train_templates=cfg["data"].get("n_train_templates"),
        n_holdout_templates=cfg["data"].get("n_holdout_templates"),
        forms=tuple(cfg["data"].get("forms", ("name", "pronoun"))),
        extra_distractors=cfg["data"].get("extra_distractors", True),
    )
    path = _data_path(cfg)
    ds.save(path)
    check = ds.check_overlaps()
    print(f"saved {path}")
    print("counts:", ds.manifest["counts"])
    print("overlap check:", check)
    if not check["ok"]:
        raise SystemExit("split overlap check FAILED")


def cmd_baseline(args):
    cfg = load_config(args.config, args.variant)
    data = IOIDataset.load(_data_path(cfg))
    device = getattr(args, "device", None) or cfg.get("device", "cuda")
    backbone = _load_backbone(cfg, device)
    from questions.q1_private_communication.src.calibration import calibrate

    out = calibrate(data, backbone, device=device, out_path=_circuit_path(cfg))
    print(json.dumps({
        "baseline_original": out["baseline_original"],
        "selected_a": out["selected_a"],
        "selected_b": out["selected_b"],
    }, indent=2, ensure_ascii=False))


def _resolve(cfg, circuit, regime):
    return resolve_config(cfg, circuit)


def cmd_train(args):
    cfg = load_config(args.config, args.variant)
    data = IOIDataset.load(_data_path(cfg))
    circuit = _load_circuit(_circuit_path(cfg))
    device = getattr(args, "device", None) or cfg.get("device", "cuda")
    backbone = _load_backbone(cfg, device)
    regimes = [getattr(args, "regime", None)] if getattr(args, "regime", None) else cfg["budget_regimes"]
    from shared.models import BudgetConfig
    from questions.q1_private_communication.src.train import (
        TrainingConfig,
        run_training_suite,
    )

    for regime in regimes:
        rd = RunDirectory(RESULTS_DIR, tag=f"{cfg['run_tag']}-{regime}")
        rd.make()
        resolved = resolve_config(cfg, circuit)
        resolved["budget_regime_run"] = regime
        rd.save_resolved_config(resolved)
        save_json(record_environment(), rd.sub("environment.json"))
        save_json(load_json(_circuit_path(cfg)), rd.sub("selected_circuit.json"))
        budget = BudgetConfig(regime=regime, **cfg["budget"][regime])
        base = TrainingConfig(
            architecture_mode="private_q",
            budget=budget,
            lr=cfg["train"]["lr"],
            steps=cfg["train"]["steps"],
            batch_size=cfg["train"]["batch_size"],
            eval_every=cfg["train"]["eval_every"],
            early_stop_patience=cfg["train"]["early_stop_patience"],
            lambda_kl=cfg["train"]["lambda_kl"],
            kl_temperature=cfg["train"]["kl_temperature"],
            seed=cfg["seed"],
            device=device,
            grad_clip=cfg["train"]["grad_clip"],
            max_eval_batches=cfg["train"].get("max_eval_batches"),
            selection_metric=cfg["train"]["selection_metric"],
        )
        report = run_training_suite(
            backbone, circuit, data, cfg["architecture_modes"], budget, base,
            rd.path, seeds=tuple(cfg["train"]["seeds"]),
        )
        _save_latest(regime, rd)
        print(json.dumps({
            "run_id": rd.run_id, "regime": regime,
            "groups": [
                {k: g.get(k) for k in ("mode", "seed", "best_val_metric", "steps_done",
                                       "wall_clock_s", "param_stats", "not_trained")}
                for g in report["groups"]
            ],
        }, indent=2, default=str))


def cmd_evaluate(args):
    cfg = load_config(args.config, args.variant)
    data = IOIDataset.load(_data_path(cfg))
    circuit = _load_circuit(_circuit_path(cfg))
    device = getattr(args, "device", None) or cfg.get("device", "cuda")
    backbone = _load_backbone(cfg, device)
    from questions.q1_private_communication.src.evaluate import run_evaluation_suite

    for regime in cfg["budget_regimes"]:
        latest = _load_latest(regime)
        if latest is None:
            print(f"no training run for regime {regime}; skipped")
            continue
        rd = RunDirectory(RESULTS_DIR, tag=f"{cfg['run_tag']}-{regime}")
        # reuse the training run directory: create a symlink-free copy of id
        rd.path = Path(latest["path"])
        rd.run_id = Path(latest["path"]).name
        summary = load_json(Path(latest["path"]) / "statistics" / "training_summary.json")
        run_evaluation_suite(
            backbone, circuit, data, cfg["architecture_modes"],
            tuple(cfg["train"]["seeds"]), summary, cfg["evaluate"]["splits"],
            rd.path, device, batch_size=cfg["evaluate"]["batch_size"],
        )
        print(f"evaluation written to {rd.path / 'statistics' / 'evaluation.json'}")


def cmd_causal(args):
    cfg = load_config(args.config, args.variant)
    data = IOIDataset.load(_data_path(cfg))
    circuit = _load_circuit(_circuit_path(cfg))
    device = getattr(args, "device", None) or cfg.get("device", "cuda")
    backbone = _load_backbone(cfg, device)
    from questions.q1_private_communication.src.causal_tests import (
        collect_measurements,
        run_causal_battery,
    )
    from questions.q1_private_communication.src.evaluate import build_trained_model

    for regime in cfg["budget_regimes"]:
        latest = _load_latest(regime)
        if latest is None:
            continue
        rd_path = Path(latest["path"])
        # The causal battery (spec 10/18) is defined for the three private
        # architectures; public_adapter and constant_* groups are evaluated
        # (mean D etc.) but have no private-channel interventions.
        private_modes = [m for m in cfg["architecture_modes"]
                         if m in ("private_q", "private_v", "private_qv")]
        for seed in cfg["train"]["seeds"]:
            for mode in private_modes:
                ckpt = rd_path / "checkpoints" / f"{mode}-seed{seed}.pt"
                if not ckpt.exists():
                    print(f"no checkpoint: {ckpt}; skipped")
                    continue
                model = build_trained_model(backbone, circuit, mode, str(ckpt))
                battery = run_causal_battery(
                    model, data, cfg["causal"]["split"], seed, device,
                    batch_size=cfg["causal"]["batch_size"],
                    n_pairs_cap=cfg["causal"].get("n_pairs_cap"),
                )
                save_json(
                    battery,
                    rd_path / "statistics" / f"causal-{mode}-seed{seed}-{regime}.json",
                )
                meas = collect_measurements(
                    model, data, cfg["causal"]["split"], device,
                    batch_size=cfg["causal"]["batch_size"],
                )
                save_json(
                    meas,
                    rd_path / "statistics" / f"measurements-{mode}-seed{seed}-{regime}.json",
                )
                print(f"causal+measurements: {mode} seed{seed} {regime}")


def cmd_report(args):
    cfg = load_config(args.config, args.variant)
    for regime in cfg["budget_regimes"]:
        latest = _load_latest(regime)
        if latest is None:
            continue
        from questions.q1_private_communication.src.reporting import build_report

        build_report(Path(latest["path"]), cfg, regime)


def cmd_smoke(args):
    cmd_data(args)
    cmd_baseline(args)
    cmd_train(args)
    cmd_evaluate(args)
    cmd_causal(args)
    cmd_report(args)
    print("smoke pipeline finished")


def _latest_file() -> Path:
    return DATA_DIR / "latest_runs.json"


def _save_latest(regime, rd: RunDirectory):
    latest = load_json(_latest_file()) if _latest_file().exists() else {}
    latest[regime] = {"run_id": rd.run_id, "path": str(rd.path),
                      "config": getattr(rd, "tag", "")}
    save_json(latest, _latest_file())


def _load_latest(regime):
    if not _latest_file().exists():
        return None
    return load_json(_latest_file()).get(regime)


def main(argv=None):
    p = argparse.ArgumentParser(prog="q1")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, help_ in (
        ("env", cmd_env, "record environment"),
        ("test", cmd_test, "run shared + Q1 tests"),
        ("data", cmd_data, "generate and check Q1 data"),
        ("baseline", cmd_baseline, "baseline + calibration"),
        ("train", cmd_train, "train all architecture groups"),
        ("evaluate", cmd_evaluate, "generalization + recovery"),
        ("causal", cmd_causal, "causal battery + measurements"),
        ("report", cmd_report, "assemble report.md"),
        ("smoke", cmd_smoke, "one-command smoke pipeline"),
    ):
        sp = sub.add_parser(name, help=help_)
        if name not in ("env", "test"):
            sp.add_argument("--config", default="smoke.yaml")
            sp.add_argument("--variant", default=None)
        if name in ("baseline", "train", "evaluate", "causal"):
            sp.add_argument("--device", default=None)
        if name == "train":
            sp.add_argument("--regime", default=None,
                            choices=["fixed_total_budget", "fixed_branch_capacity"])
        sp.set_defaults(fn=fn)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

"""Q1 test 8 (spec 12): small end-to-end smoke pipeline (data -> train 1
group for a few steps -> evaluate -> causal battery) runs on CPU."""

import tempfile
from pathlib import Path

import torch

from shared.models import BudgetConfig, BudgetRanks, CircuitSpec, PrivateTransformer
from tests.shared.conftest import make_model
from questions.q1_private_communication.src.data import generate_ioi
from questions.q1_private_communication.src.train import TrainingConfig, train_group
from questions.q1_private_communication.src.causal_tests import run_causal_battery
from questions.q1_private_communication.src.evaluate import evaluate_group


def test_smoke_pipeline(backbone, tiny_dataset):
    circuit = CircuitSpec(
        a_heads=tuple(tiny_dataset_heads_a()),
        b_heads=tuple(tiny_dataset_heads_b()),
    )
    cfg = TrainingConfig(
        architecture_mode="private_q",
        budget=BudgetConfig(regime="fixed_branch_capacity", branch_rank=2),
        lr=1e-3,
        steps=3,
        batch_size=8,
        eval_every=3,
        early_stop_patience=2,
        seed=0,
        device="cpu",
    )
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        res = train_group(backbone, circuit, cfg, tiny_dataset, run_dir)
        assert res.steps_done == 3
        assert res.best_checkpoint is not None
        # evaluation on one split
        ev = evaluate_group(
            backbone, circuit, "private_q", 0, tiny_dataset,
            ["test_combination"], res.best_checkpoint, run_dir, device="cpu",
            batch_size=8,
        )
        assert ev["splits"]["test_combination"]["overall"]["n"] > 0
        # causal battery (small pair cap)
        model = PrivateTransformer(backbone, circuit, "private_q",
                                   BudgetRanks(q_rank=2), seed=0)
        from shared.training.checkpoint import load_private_checkpoint

        load_private_checkpoint(model, res.best_checkpoint)
        battery = run_causal_battery(
            model, tiny_dataset, "test_combination", 0, device="cpu",
            batch_size=8, n_pairs_cap=4,
        )
        assert battery["tests"]["off"]["equals_delete"] is True
        assert len(battery["records"]["paired_swap"]) > 0
        assert all(r["donor_id"] is not None for r in battery["records"]["paired_swap"])


def tiny_dataset_heads_a():
    from shared.models import HeadRef

    return [HeadRef(7, 3), HeadRef(8, 6)]


def tiny_dataset_heads_b():
    from shared.models import HeadRef

    return [HeadRef(9, 6), HeadRef(10, 0)]

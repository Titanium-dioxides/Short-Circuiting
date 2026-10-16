"""Integrity tests (spec 12 tests 9-13; spec 20 tests 9, 11-13):
config validation, cache isolation, freezing/gradients/checkpointing,
batch determinism, single-forward guarantee, constant-mode semantics and
parameter-budget accounting."""

import pytest
import torch

from shared.models import (
    BudgetConfig,
    BudgetRanks,
    CircuitSpec,
    HeadRef,
    Intervention,
    PrivateTransformer,
    param_counts_for_mode,
    validate_budget_ranks,
)
from shared.models.backbone import param_fingerprint
from shared.training.checkpoint import (
    load_private_checkpoint,
    save_private_checkpoint,
    verify_backbone_frozen,
    verify_gradients,
)
from tests.shared.conftest import TOKENS, make_model

P = TOKENS.shape[1] - 1


def _randomize(m):
    with torch.no_grad():
        for p in m.adapters.parameters():
            p.add_(torch.randn_like(p) * 0.5)


# ---------------------------------------------------------------- spec 12.9


def test_illegal_routing_configs_raise(backbone):
    # A.layer >= B.layer must raise (any violating pair)
    bad = CircuitSpec(a_heads=(HeadRef(9, 6),), b_heads=(HeadRef(9, 0),))
    with pytest.raises(ValueError, match="illegal connection"):
        PrivateTransformer(backbone, bad, "private_q", BudgetRanks(q_rank=2))
    # a violating pair in a larger set raises the same per-pair error
    bad2 = CircuitSpec(a_heads=(HeadRef(7, 3), HeadRef(9, 6)), b_heads=(HeadRef(9, 0),))
    with pytest.raises(ValueError, match="illegal connection"):
        PrivateTransformer(backbone, bad2, "private_q", BudgetRanks(q_rank=2))
    # empty sets
    with pytest.raises(ValueError):
        CircuitSpec(a_heads=(), b_heads=(HeadRef(9, 0),)).validate()
    with pytest.raises(ValueError):
        CircuitSpec(a_heads=(HeadRef(7, 3),), b_heads=()).validate()


# ---------------------------------------------------------------- spec 12.10


def test_cache_no_cross_forward_pollution(backbone, circuit):
    m = make_model(backbone, circuit, "private_qv")
    _randomize(m)
    with torch.no_grad():
        _, state1 = m.forward(TOKENS, need_analysis=True)
        s1 = state1.s_B.clone()
        _, state2 = m.forward(torch.flip(TOKENS, dims=[0]), need_analysis=True)
        s2 = state2.s_B.clone()
    # second forward's cache is its own; flipping inputs must flip the signal
    assert state1 is not state2
    assert not torch.allclose(s1, s2, atol=1e-6)
    assert torch.allclose(s2, torch.flip(s1, dims=[0]), atol=1e-6)
    # no lingering fwd/bwd hooks attached to any hook point after forwards
    for hp in backbone.hook_dict.values():
        assert len(hp.fwd_hooks) == 0 and len(hp.bwd_hooks) == 0


# ---------------------------------------------------------------- spec 12.11 / 20.9


def test_frozen_backbone_and_adapter_gradients(backbone, circuit):
    fp0 = param_fingerprint(backbone)
    m = make_model(backbone, circuit, "private_qv")
    # W_down gradients are zero BY CONSTRUCTION at zero init (they only flow
    # through W_up = 0); randomize first so the strict gradient check is
    # meaningful (spec 9: valid gradients during training).
    _randomize(m)
    report = verify_gradients(m, TOKENS)
    assert report["backbone_has_grad"] is False
    assert len(report["branches"]) == 2 * circuit.n_b * 3  # q+v, 3 params each
    assert all(v > 0 for v in report["branches"].values())
    assert verify_backbone_frozen(m, fp0)
    # one optimizer step on adapters only must not touch the backbone
    opt = torch.optim.Adam(m.trainable_parameters(), lr=1e-3)
    logits, _ = m.forward(TOKENS)
    loss = logits[:, P, :].pow(2).mean()
    loss.backward()
    opt.step()
    assert verify_backbone_frozen(m, fp0)


# ---------------------------------------------------------------- spec 12.12


def test_batch_reorder_does_not_change_predictions(backbone, circuit):
    m = make_model(backbone, circuit, "private_qv")
    _randomize(m)
    idx = torch.tensor([1, 0])
    with torch.no_grad():
        l_ab, _ = m.forward(TOKENS)
        l_ba, _ = m.forward(TOKENS[idx])
    assert torch.allclose(l_ba, l_ab[idx], atol=1e-6, rtol=1e-6)


def test_checkpoint_restore_does_not_change_predictions(backbone, circuit, tmp_path):
    m = make_model(backbone, circuit, "private_qv", seed=3)
    _randomize(m)
    fp0 = param_fingerprint(backbone)
    with torch.no_grad():
        before, _ = m.forward(TOKENS)
    path = tmp_path / "ckpt.pt"
    save_private_checkpoint(m, path, meta={"step": 10})
    m2 = make_model(backbone, circuit, "private_qv", seed=3)
    load_private_checkpoint(m2, path, ref_fingerprint=fp0)
    with torch.no_grad():
        after, _ = m2.forward(TOKENS)
    assert torch.allclose(before, after, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------- spec 12.13


def test_private_forward_needs_no_extra_clean_forward(backbone, circuit):
    calls = []
    orig_forward = backbone.forward

    def counting_forward(*args, **kwargs):
        calls.append(1)
        return orig_forward(*args, **kwargs)

    backbone.forward = counting_forward  # type: ignore[method-assign]
    try:
        m = make_model(backbone, circuit, "private_qv")
        _randomize(m)
        with torch.no_grad():
            m.forward(TOKENS)
            m.forward(TOKENS, intervention=Intervention(name="off"))
            m.forward(TOKENS, intervention=Intervention(name="mean_input",
                                                        mean_signal=torch.zeros(1, m.in_dim)))
            don = m.collect_donor_signal(TOKENS)
            m.forward(TOKENS, intervention=Intervention(name="paired_swap", donor_signal=don))
    finally:
        backbone.forward = orig_forward  # type: ignore[method-assign]
    # one forward per call; donor collection is an explicit separate external
    # run (spec 10.5), so 4 + 1 = 5 total
    assert len(calls) == 5


# ---------------------------------------------------------------- spec 20.11


def test_constant_modes_semantics(backbone, circuit):
    for mode, key in (("constant_q", "delta_q"), ("constant_v", "gamma"),
                      ("constant_qv", "delta_q")):
        m = make_model(backbone, circuit, mode)
        with torch.no_grad():
            logits0, _ = m.forward(TOKENS)
            logits_del, _ = make_model(backbone, circuit, "delete").forward(TOKENS)
        assert torch.allclose(logits0, logits_del, atol=1e-5), f"{mode} zero-init"
        _randomize(m)
        with torch.no_grad():
            logits1, state = m.forward(TOKENS, need_analysis=True)
            # sample-independent: identical modulation across the batch
            vals = list(state.delta_q.values()) + list(state.gamma.values())
            for v in vals:
                assert torch.allclose(v[0], v[1], atol=1e-6), (
                    f"{mode}: modulation must be sample-independent"
                )
            # off zeroes the constant modulation -> equals delete again
            logits_off, _ = m.forward(TOKENS, intervention=Intervention(name="off"))
            assert torch.allclose(logits_off, logits_del, atol=1e-5)
            assert not torch.allclose(logits1, logits_del, atol=1e-7), mode
        # constant modes have no signal: sample-dependent interventions raise
        with pytest.raises(ValueError):
            m.forward(TOKENS, intervention=Intervention(name="mean_input",
                                                        mean_signal=torch.zeros(1, m.in_dim)))
        with pytest.raises(ValueError):
            m.forward(TOKENS, intervention=Intervention(
                name="paired_swap", donor_signal=torch.zeros(2, m.in_dim)))
    # constant_qv: both branches present and independent
    m = make_model(backbone, circuit, "constant_qv")
    assert m.adapters.q.b_heads == m.adapters.v.b_heads


# ---------------------------------------------------------------- spec 20.12


def test_checkpoint_saves_branches_and_mode(backbone, circuit, tmp_path):
    m = make_model(backbone, circuit, "private_qv", seed=5)
    _randomize(m)
    path = tmp_path / "qv.pt"
    save_private_checkpoint(m, path, meta={"architecture_mode": m.mode})
    m2 = make_model(backbone, circuit, "private_qv", seed=5)
    meta = load_private_checkpoint(m2, path)
    assert meta["architecture_mode"] == "private_qv"
    assert meta["ranks"] == {"q_rank": 4, "v_rank": 4, "pub_rank": 4}
    with torch.no_grad():
        l1, _ = m.forward(TOKENS)
        l2, _ = m2.forward(TOKENS)
    assert torch.allclose(l1, l2, atol=1e-6)
    # mode mismatch must raise
    m3 = make_model(backbone, circuit, "private_q", seed=5)
    with pytest.raises(ValueError):
        load_private_checkpoint(m3, path)


# ---------------------------------------------------------------- spec 20.13


def test_param_budget_stats_and_config_validation():
    in_dim, d_head, d_model, n_b = 1536, 64, 768, 2

    # fixed branch capacity: qv must have strictly more params than q/v
    cfg = BudgetConfig(regime="fixed_branch_capacity", branch_rank=4)
    table = validate_budget_ranks(cfg, in_dim, d_head, d_model, n_b)
    assert table["private_q"]["total"] == table["private_v"]["total"]
    assert table["private_qv"]["total"] > table["private_q"]["total"]
    assert table["private_qv"]["q_branch"] == table["private_qv"]["v_branch"]
    # per-head LowRankAffine count: r*in + out*r + out
    expected_per_head = 4 * 1536 + 64 * 4 + 64
    assert table["private_q"]["q_branch"] == expected_per_head * n_b
    assert table["constant_q"]["constants"] == n_b * d_head
    assert table["constant_qv"]["constants"] == 2 * n_b * d_head
    # public_adapter sized to private_qv actual total: the closest rank that
    # does not exceed it (exact match not always possible); the gap must be
    # smaller than one rank step (in_dim + d_model params)
    pub, qv = table["public_adapter"]["total"], table["private_qv"]["total"]
    assert pub <= qv
    assert qv - pub < in_dim + d_model + 1

    # fixed total budget: qv splits equally, all within budget, deviations
    # recorded, no padding params
    P = 20000
    cfg2 = BudgetConfig(regime="fixed_total_budget", total_params=P)
    table2 = validate_budget_ranks(cfg2, in_dim, d_head, d_model, n_b)
    assert table2["private_qv"]["q_branch"] == table2["private_qv"]["v_branch"]
    assert table2["private_qv"]["total"] <= P
    assert table2["private_qv"]["deviation_from_budget"] == table2["private_qv"]["total"] - P
    assert table2["private_q"]["total"] <= P
    assert table2["private_q"]["ranks"]["q_rank"] >= 1
    assert table2["private_v"]["ranks"]["v_rank"] >= 1
    assert table2["private_qv"]["ranks"]["q_rank"] >= 1
    assert table2["private_qv"]["ranks"]["v_rank"] >= 1

    # invalid configs raise
    with pytest.raises(ValueError):
        validate_budget_ranks(BudgetConfig(regime="fixed_branch_capacity"), in_dim, d_head, d_model, n_b)
    with pytest.raises(ValueError):
        validate_budget_ranks(BudgetConfig(regime="fixed_total_budget", total_params=10),
                              in_dim, d_head, d_model, n_b)
    with pytest.raises(ValueError):
        validate_budget_ranks(BudgetConfig(regime="fixed_total_budget", total_params=0),
                              in_dim, d_head, d_model, n_b)

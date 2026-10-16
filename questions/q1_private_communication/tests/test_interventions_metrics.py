"""Q1 tests 6-7 (spec 12): mean/shuffle/swap intervention semantics and
metrics/recovery-rate edge cases."""

import pytest
import torch

from shared.models import BudgetRanks, Intervention, PrivateTransformer
from shared.evaluation.statistics import (
    DENOMINATOR_MIN,
    bootstrap_paired_ci,
    recovery_rate,
)
from tests.shared.conftest import TOKENS, make_model


def _randomize(m):
    with torch.no_grad():
        for p in m.adapters.parameters():
            p.add_(torch.randn_like(p) * 0.5)


def test_mean_input_semantics(backbone, circuit):
    m = make_model(backbone, circuit, "private_q")
    _randomize(m)
    with torch.no_grad():
        _, state = m.forward(TOKENS, intervention=Intervention(name="normal"),
                             force_signal=True)
        true_mean = state.s_B.detach().mean(dim=0, keepdim=True)
        est = m.estimate_mean_signal(TOKENS)
    assert torch.allclose(est, true_mean, atol=1e-6)
    # mean_input forward: delta_q equals adapter(mean), not adapter(s_B)
    with torch.no_grad():
        _, st = m.forward(
            TOKENS,
            intervention=Intervention(name="mean_input", mean_signal=est),
            need_analysis=True,
        )
        for key, delta in st.delta_q.items():
            layer = int(key.split("H")[0][1:])
            head = int(key.split("H")[1])
            from shared.models import HeadRef

            expected = m.adapters.delta(HeadRef(layer, head), est)
            assert torch.allclose(delta, expected, atol=1e-5)


def test_shuffled_input_semantics(backbone, circuit):
    """shuffled_input: each recipient row receives the permuted row's whole
    signal group; the applied delta_q equals adapter(s_B[perm]) — only the
    signal rows move, the recipient's own computation stays put."""
    from shared.models import HeadRef

    m = make_model(backbone, circuit, "private_q")
    _randomize(m)
    perm = torch.tensor([1, 0])
    with torch.no_grad():
        _, st = m.forward(
            TOKENS,
            intervention=Intervention(name="shuffled_input", shuffle_perm=perm),
            force_signal=True, need_analysis=True,
        )
        assert st.s_B is not None
        s_perm = st.s_B.detach()[perm]
        for key, delta in st.delta_q.items():
            layer = int(key.split("H")[0][1:])
            head = int(key.split("H")[1])
            expected = m.adapters.delta(HeadRef(layer, head), s_perm)
            assert torch.allclose(delta, expected, atol=1e-5)
    # and the shuffled run differs from the normal run (signals moved)
    with torch.no_grad():
        l_sh, _ = m.forward(TOKENS, intervention=Intervention(
            name="shuffled_input", shuffle_perm=perm))
        l_norm, _ = m.forward(TOKENS)
    assert not torch.allclose(l_sh, l_norm, atol=1e-5)


def test_donor_swap_semantics(backbone, circuit):
    """paired_swap: only the private channel input is replaced; the delta_q
    applied equals adapter(donor_signal), and the recipient's residual/K/V
    path is its own (external-donor diagnostic)."""
    m = make_model(backbone, circuit, "private_q")
    _randomize(m)
    with torch.no_grad():
        donor = m.collect_donor_signal(TOKENS[1:2])  # donor = sample 1
        _, st = m.forward(
            TOKENS[0:1],
            intervention=Intervention(name="paired_swap", donor_signal=donor),
            need_analysis=True,
        )
        for key, delta in st.delta_q.items():
            layer = int(key.split("H")[0][1:])
            head = int(key.split("H")[1])
            from shared.models import HeadRef

            expected = m.adapters.delta(HeadRef(layer, head), donor)
            assert torch.allclose(delta, expected, atol=1e-5)
        # donor collection runs are detached
        assert not donor.requires_grad


def test_metrics_and_recovery_edge_cases():
    # denominator <= 0 -> not interpretable
    r, ok, reason = recovery_rate(1.0, 2.0, 1.5)
    assert r is None and not ok and "<= 0" in reason
    # denominator < DENOMINATOR_MIN -> not interpretable
    r, ok, reason = recovery_rate(1.0, 1.0 - DENOMINATOR_MIN / 2, 1.0)
    assert r is None and not ok and "too small" in reason
    # normal case: R = ratio of means
    r, ok, reason = recovery_rate(2.0, 0.5, 1.25)
    assert ok and abs(r - 0.5) < 1e-9
    # paired bootstrap: resampling unit is pair_id
    ids = ["p1", "p1", "p2", "p2"]
    values = {"p1": 1.0, "p2": 2.0}
    res = bootstrap_paired_ci(ids, lambda sel: sum(values[i] for i in sel) / len(sel),
                              n_boot=500, seed=0)
    assert abs(res["estimate"] - 1.5) < 1e-9
    assert res["ci_low"] >= 1.0 - 1e-9 and res["ci_high"] <= 2.0 + 1e-9

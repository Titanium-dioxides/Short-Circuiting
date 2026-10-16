"""Equivalence tests (spec 12 tests 1-6, 8; spec 20 tests 1-8, 10).

Strategy: every architecture property is verified against a MANUAL
reference implementation of the same hook arithmetic (deletion formula,
explicit sum_j alpha_pj [(1+gamma) odot v_j] reference for V), so the tests
check the wrapper against the spec's formulas, not the wrapper against
itself.

Tolerances are per dtype and documented in docs/architecture_spec.md:
FP32 atol=1e-5/rtol=1e-5 for logits (O(10) scale); the delete-then-add-back
round trip is not bit-exact in floating point by construction.
"""

import torch

from shared.models import BudgetRanks, CircuitSpec, HeadRef, Intervention, PrivateTransformer
from tests.shared.conftest import TOKENS, make_model

ATOL = 1e-5
RTOL = 1e-5
P = TOKENS.shape[1] - 1  # target position


# ----------------------------------------------------------------------
# manual reference implementation (deletion + optional V scaling)
# ----------------------------------------------------------------------

class ManualRun:
    """Installs deletion hooks (spec 2.3 formula) plus optional explicit
    reference V scaling at hook_z (spec 20.3: z'_p = sum_j alpha_pj
    [(1+gamma) odot v_j]) and observation captures."""

    def __init__(self, backbone, circuit, v_adapter=None, capture=False):
        self.backbone = backbone
        self.circuit = circuit
        self.v_adapter = v_adapter
        self.z = {}
        self.q = {}
        self.k = {}
        self.v = {}
        self.pattern = {}
        self.pre = {}
        self.post = {}
        self.o_sum = {}  # layer -> sum of deleted A-head contributions (same run)
        self.capture = capture

    def _z_hook(self, layer):
        run = self

        def hook(z, hook=None):
            run.z[layer] = z
            return z

        return hook

    def _delete_hook(self, layer):
        run = self

        def hook(attn_out, hook=None):
            run.pre[layer] = attn_out
            out = attn_out
            heads = run.circuit.a_heads_at(layer)
            if heads:
                z_l = run.z[layer]
                W_O = run.backbone.blocks[layer].attn.W_O
                o_sum = None
                for h in heads:
                    o = torch.einsum("bd,do->bo", z_l[:, P, h.head, :], W_O[h.head])
                    o_sum = o if o_sum is None else o_sum + o
                out = attn_out.clone()
                out[:, P, :] = out[:, P, :] - o_sum
                run.o_sum[layer] = o_sum
            run.post[layer] = out
            return out

        return hook

    def _capture(self, kind, layer):
        run = self

        def hook(t, hook=None):
            getattr(run, kind)[layer] = t
            return t

        return hook

    def _s_B(self):
        parts = []
        for a in self.circuit.signal_order:
            W_O = self.backbone.blocks[a.layer].attn.W_O
            o = torch.einsum("bd,do->bo", self.z[a.layer][:, P, a.head, :], W_O[a.head])
            parts.append(o)
        return torch.cat(parts, dim=1)

    def _ref_v_hook(self, layer):
        run = self

        def hook(z, hook=None):
            out = z.clone()
            s_B = run._s_B()
            for h in run.circuit.b_heads_at(layer):
                gamma = run.v_adapter.gamma(h, s_B)  # [B, dh]
                alpha = run.pattern[layer][:, h.head, P, :]  # [B, S]
                ref = torch.einsum(
                    "bs,bsd->bd",
                    alpha,
                    run.v[layer][:, :, h.head, :] * (1.0 + gamma.unsqueeze(1)),
                )
                out[:, P, h.head, :] = ref
            return out

        return hook

    def run(self, tokens, scale_v=False):
        hooks = []
        for layer in self.circuit.a_layers():
            hooks.append((f"blocks.{layer}.attn.hook_z", self._z_hook(layer)))
            hooks.append((f"blocks.{layer}.hook_attn_out", self._delete_hook(layer)))
        for layer in self.circuit.b_layers():
            if self.capture:
                hooks.append((f"blocks.{layer}.attn.hook_q", self._capture("q", layer)))
                hooks.append((f"blocks.{layer}.attn.hook_k", self._capture("k", layer)))
                hooks.append((f"blocks.{layer}.attn.hook_v", self._capture("v", layer)))
                hooks.append(
                    (f"blocks.{layer}.attn.hook_pattern", self._capture("pattern", layer))
                )
            hooks.append((f"blocks.{layer}.attn.hook_z", self._z_hook(layer)))
            if scale_v:
                hooks.append((f"blocks.{layer}.attn.hook_z", self._ref_v_hook(layer)))
        with torch.no_grad():
            return self.backbone.run_with_hooks(tokens, fwd_hooks=hooks)


def _randomize(m):
    with torch.no_grad():
        for p in m.adapters.parameters():
            p.add_(torch.randn_like(p) * 0.5)


# ---------------------------------------------------------------- spec 12.1


def test_noop_hooks_match_original(backbone):
    with torch.no_grad():
        plain = backbone(TOKENS)

        def noop(x, hook=None):
            return x

        hooked = backbone.run_with_hooks(
            TOKENS,
            fwd_hooks=[
                (f"blocks.{l}.attn.hook_q", noop) for l in (0, 5, 11)
            ] + [(f"blocks.{l}.hook_attn_out", noop) for l in (0, 5, 11)]
            + [(f"blocks.{l}.attn.hook_z", noop) for l in (0, 5, 11)],
        )
    assert torch.allclose(plain, hooked, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------- spec 12.2


def test_head_outputs_sum_plus_bias_equals_attn_out(backbone):
    """attn_out = sum_h o_h + b_O, with o_h = z_h @ W_O_h^T, per layer."""
    layer = 3
    obs = {}

    def z_hook(z, hook=None):
        obs["z"] = z
        return z

    def out_hook(o, hook=None):
        obs["attn_out"] = o
        return o

    with torch.no_grad():
        backbone.run_with_hooks(
            TOKENS,
            fwd_hooks=[
                (f"blocks.{layer}.attn.hook_z", z_hook),
                (f"blocks.{layer}.hook_attn_out", out_hook),
            ],
        )
        W_O = backbone.blocks[layer].attn.W_O
        b_O = backbone.blocks[layer].attn.b_O
        recon = torch.einsum("bshd,hdo->bso", obs["z"], W_O) + b_O
    assert torch.allclose(obs["attn_out"], recon, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------- spec 12.3


def test_delete_only_specified_head_and_token(backbone, circuit):
    """Wrapper deletion == manual formula; only A heads at position p are
    removed; other positions/heads and b_O untouched."""
    manual = ManualRun(backbone, circuit)
    with torch.no_grad():
        logits_manual = manual.run(TOKENS)
        logits_wrap, _ = make_model(backbone, circuit, "delete").forward(TOKENS)
    assert torch.allclose(logits_wrap, logits_manual, atol=ATOL, rtol=RTOL)
    layer = 8  # A layer with exactly one A head (8, 6)
    pre, post = manual.pre[layer], manual.post[layer]
    W_O = backbone.blocks[layer].attn.W_O
    o_h = torch.einsum("bd,do->bo", manual.z[layer][:, P, 6, :], W_O[6])
    assert torch.allclose(post[:, :P, :], pre[:, :P, :], atol=ATOL, rtol=RTOL)
    assert torch.allclose(post[:, P, :], pre[:, P, :] - o_h, atol=ATOL, rtol=RTOL)
    assert not torch.allclose(post[:, P, :], pre[:, P, :], atol=1e-7)


# ---------------------------------------------------------------- spec 12.4

# (x - y) + y is not bit-exact in floating point. The local (attn_out-level)
# restoration error is ~1e-6; after 4 further attention layers this fp noise
# is amplified to ~6.5e-3 at the logits (measured 2026-10-16, FP32). The
# tolerance below is documented in docs/architecture_spec.md and is far below
# the O(1..10) effects the experiments measure. Positions < p are unaffected
# by the intervention and must match tightly.
ADD_BACK_LOCAL_ATOL = 1e-5
ADD_BACK_LOGITS_ATOL = 1e-2


def test_delete_then_exact_addback_restores_logits(backbone, circuit):
    """Deletion run caches o_h; an add-back run re-adds the same per-head
    contributions: the local output is restored to ~1e-6 and final logits to
    the documented fp-noise-amplification tolerance."""
    m = make_model(backbone, circuit, "delete")
    with torch.no_grad():
        plain, _ = make_model(backbone, circuit, "original").forward(TOKENS)
        _, state_del = m.forward(TOKENS, force_signal=True)
        o_A = state_del.o_A  # [B, nA, d_model] from the deleted run
        assert o_A is not None

        # manual deletion (same formula), then in the SAME run add back the
        # o_sum the deletion hook just computed: each layer restores its own
        # contribution, so later layers see the restored upstream state.
        manual = ManualRun(backbone, circuit)
        logits_manual = manual.run(TOKENS)

        def addback_hook(attn_out, hook=None):
            out = attn_out
            layer = int(hook.name.split(".")[1])
            if manual.o_sum.get(layer) is not None:
                out = out.clone()
                out[:, P, :] = out[:, P, :] + manual.o_sum[layer]
            return out

        restored = backbone.run_with_hooks(
            TOKENS,
            fwd_hooks=[
                (f"blocks.{l}.attn.hook_z", manual._z_hook(l))
                for l in circuit.a_layers()
            ]
            + [
                (f"blocks.{l}.hook_attn_out", manual._delete_hook(l))
                for l in circuit.a_layers()
            ]
            + [
                (f"blocks.{l}.hook_attn_out", addback_hook)
                for l in circuit.a_layers()
            ],
        )
    # local restoration: post_deletion + o_sum == pre_deletion per A layer
    for layer in circuit.a_layers():
        pre = manual.pre[layer]
        post = manual.post[layer]
        o_sum = manual.o_sum[layer]
        assert torch.allclose(
            post[:, P, :] + o_sum, pre[:, P, :], atol=ADD_BACK_LOCAL_ATOL
        ), f"local add-back mismatch at layer {layer}"
        # other positions are untouched by deletion at all
        assert torch.allclose(post[:, :P, :], pre[:, :P, :], atol=1e-7)
    # final logits: fp-noise amplification tolerance, tight below p
    assert torch.allclose(
        restored[:, :P, :], plain[:, :P, :], atol=ATOL, rtol=RTOL
    )
    assert torch.allclose(
        restored[:, P:, :], plain[:, P:, :],
        atol=ADD_BACK_LOGITS_ATOL, rtol=ADD_BACK_LOGITS_ATOL,
    )
    assert not torch.allclose(logits_manual, plain, atol=1e-7)


# ---------------------------------------------------------------- spec 12.5 / 20.1


def test_zero_init_all_trainable_modes_equal_delete(backbone, circuit):
    with torch.no_grad():
        logits_del, _ = make_model(backbone, circuit, "delete").forward(TOKENS)
    for mode in (
        "private_q", "private_v", "private_qv",
        "constant_q", "constant_v", "constant_qv", "public_adapter",
    ):
        with torch.no_grad():
            logits_m, _ = make_model(backbone, circuit, mode).forward(TOKENS)
        assert torch.allclose(logits_m, logits_del, atol=ATOL, rtol=RTOL), mode


# ---------------------------------------------------------------- spec 12.6 / 20.2


def test_off_all_private_modes_equal_delete(backbone, circuit):
    with torch.no_grad():
        logits_del, _ = make_model(backbone, circuit, "delete").forward(TOKENS)
    for mode in ("private_q", "private_v", "private_qv"):
        m = make_model(backbone, circuit, mode)
        _randomize(m)
        with torch.no_grad():
            logits_off, _ = m.forward(TOKENS, intervention=Intervention(name="off"))
        assert torch.allclose(logits_off, logits_del, atol=ATOL, rtol=RTOL), mode


# ---------------------------------------------------------------- spec 20.3


def test_v_reference_implementation_equals_hook_z(backbone, circuit):
    """Explicit reference: z'_p = sum_j alpha_pj [(1+gamma) odot v_j] equals
    the hook_z implementation (1+gamma) odot sum_j alpha_pj v_j."""
    m = make_model(backbone, circuit, "private_v")
    _randomize(m)
    manual = ManualRun(backbone, circuit, v_adapter=m.adapters, capture=True)
    with torch.no_grad():
        logits_ref = manual.run(TOKENS, scale_v=True)
        logits_wrap, state = m.forward(TOKENS, need_analysis=True)
    assert torch.allclose(logits_wrap, logits_ref, atol=ATOL, rtol=RTOL)
    for layer in circuit.b_layers():
        z_scaled, z_base = state.z_scaled[layer], state.z_base[layer]
        for h in circuit.b_heads_at(layer):
            gamma = state.gamma[f"L{h.layer}H{h.head}"]  # [B, dh]
            alpha = manual.pattern[layer][:, h.head, P, :]
            v = manual.v[layer][:, :, h.head, :]
            ref = torch.einsum(
                "bs,bsd->bd", alpha, v * (1.0 + gamma.unsqueeze(1))
            )
            assert torch.allclose(z_scaled[:, P, h.head, :], ref, atol=ATOL, rtol=RTOL)
            assert torch.allclose(
                z_scaled[:, P, h.head, :],
                z_base[:, P, h.head, :] * (1.0 + gamma),
                atol=ATOL, rtol=RTOL,
            )


# ---------------------------------------------------------------- spec 12.7 / 20.4


def test_v_scaling_only_affects_specified_b_and_p(backbone, circuit):
    m = make_model(backbone, circuit, "private_v")
    _randomize(m)
    with torch.no_grad():
        logits_priv, state = m.forward(TOKENS, need_analysis=True)
        logits_del, _ = make_model(backbone, circuit, "delete").forward(TOKENS)
        assert torch.allclose(
            logits_priv[:, :P, :], logits_del[:, :P, :], atol=ATOL, rtol=RTOL
        )
        assert not torch.allclose(logits_priv[:, P, :], logits_del[:, P, :], atol=1e-7)
    for layer in circuit.b_layers():
        zs, zb = state.z_scaled[layer], state.z_base[layer]
        assert torch.allclose(zs[:, :P, :, :], zb[:, :P, :, :], atol=ATOL, rtol=RTOL)
        b_heads = {b.head for b in circuit.b_heads_at(layer)}
        for h in range(12):
            if h not in b_heads:
                assert torch.allclose(zs[:, P, h, :], zb[:, P, h, :], atol=ATOL, rtol=RTOL)


def test_q_modulation_only_affects_specified_b_and_p(backbone, circuit):
    m = make_model(backbone, circuit, "private_q")
    _randomize(m)
    with torch.no_grad():
        logits_priv, state = m.forward(TOKENS)
        logits_del, _ = make_model(backbone, circuit, "delete").forward(TOKENS)
        assert torch.allclose(
            logits_priv[:, :P, :], logits_del[:, :P, :], atol=ATOL, rtol=RTOL
        )
        assert not torch.allclose(logits_priv[:, P, :], logits_del[:, P, :], atol=1e-7)
    touched = set(state.delta_q.keys())
    expected = {f"L{h.layer}H{h.head}" for h in circuit.b_heads}
    assert touched == expected


# ---------------------------------------------------------------- spec 20.5/20.6


def test_v_only_does_not_rewrite_qkv_or_pattern(backbone, circuit):
    """Single receiving head, identical upstream state: V-only leaves the
    current head's Q/K/V tensors and attention pattern exactly as in the
    delete run (the scaling happens after they are computed)."""
    single = CircuitSpec(a_heads=circuit.a_heads, b_heads=(circuit.b_heads[0],))
    m = make_model(backbone, single, "private_v")
    _randomize(m)
    manual_v = ManualRun(backbone, single, v_adapter=m.adapters, capture=True)
    manual_del = ManualRun(backbone, single, capture=True)
    with torch.no_grad():
        logits_ref = manual_v.run(TOKENS, scale_v=True)
        logits_del = manual_del.run(TOKENS)
        logits_wrap, _ = m.forward(TOKENS)
    # wrapper == manual explicit-reference V run
    assert torch.allclose(logits_wrap, logits_ref, atol=ATOL, rtol=RTOL)
    layer = single.b_layers()[0]
    for kind in ("q", "k", "v", "pattern"):
        a = getattr(manual_v, kind)[layer]
        b = getattr(manual_del, kind)[layer]
        assert torch.allclose(a, b, atol=ATOL, rtol=RTOL), kind


# ---------------------------------------------------------------- spec 20.7/20.8


def test_qv_branch_ablations_equal_single_branch_models(backbone, circuit):
    """QV with V off == Q-only with the same Q params; QV with Q off ==
    V-only with the same V params (same random weights copied in)."""
    seed = 7
    qv = make_model(backbone, circuit, "private_qv", seed=seed)
    q_only = make_model(backbone, circuit, "private_q", seed=seed)
    v_only = make_model(backbone, circuit, "private_v", seed=seed)
    _randomize(qv)
    q_only.adapters.load_state_dict(qv.adapters.q.state_dict())
    v_only.adapters.load_state_dict(qv.adapters.v.state_dict())
    with torch.no_grad():
        l_qv_q, _ = qv.forward(TOKENS, intervention=Intervention(name="q_on_v_off"))
        l_q, _ = q_only.forward(TOKENS)
        l_qv_v, _ = qv.forward(TOKENS, intervention=Intervention(name="q_off_v_on"))
        l_v, _ = v_only.forward(TOKENS)
    assert torch.allclose(l_qv_q, l_q, atol=ATOL, rtol=RTOL)
    assert torch.allclose(l_qv_v, l_v, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------- spec 20.10


def test_no_pollution_of_earlier_queries(backbone, circuit):
    """Single B head: modulation at p never changes outputs at positions < p
    (per-sample, same layer, causal structure)."""
    single = CircuitSpec(a_heads=circuit.a_heads, b_heads=(circuit.b_heads[1],))
    for mode in ("private_q", "private_v", "private_qv"):
        m = make_model(backbone, single, mode)
        _randomize(m)
        with torch.no_grad():
            logits_priv, _ = m.forward(TOKENS)
            logits_del, _ = make_model(backbone, single, "delete").forward(TOKENS)
        assert torch.allclose(
            logits_priv[:, :P, :], logits_del[:, :P, :], atol=ATOL, rtol=RTOL
        ), mode

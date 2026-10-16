"""Hook backend: per-forward A caching + public writeback deletion, Q
modulation, and post-aggregation z scaling (the exact V-only equivalent),
with hook installation, verification and cleanup.

All hooks are installed as temporary ``fwd_hooks`` for a single
``run_with_hooks`` call and are removed by TransformerLens afterwards; every
forward uses its own ``ForwardState`` (spec 2.5/2.6: independent, cleanable
per-forward caches, no cross-batch/cross-run pollution).

Deletion (spec 2.3): at layer l's ``hook_attn_out`` (post W_O + b_O, pre
residual add), for the target position p:
    attn_out_private[p] = attn_out_current_run[p] - sum_{h in A@l} o_h[p]
with o_h[p] = z_h[p] @ W_O_h^T (d_model-dim per-head contribution after W_O;
b_O is neither deleted nor re-added; other heads and positions untouched).

Q modulation (spec 2.4/15.3): at ``hook_q`` (post W_Q), q'[p] = q[p] + delta_q.

V scaling (spec 15.4/15.5): at ``hook_z`` (post aggregation, pre W_O),
z'[p] = (1 + gamma) * z[p]. This is the exact algebraic equivalent of
scaling v at every visible source position j for the target query p; it
modifies only the designated receiving head B at position p, and no Q/K/V
tensor or attention pattern is directly rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from shared.models.routing import ArchitectureMode, CircuitSpec, INTERVENTION_MODES

# ------------------------------------------------------------------- state


@dataclass
class ForwardState:
    """Independent per-forward cache. Never reused across forwards."""

    z_base: dict[int, torch.Tensor] = field(default_factory=dict)  # layer -> z (B layers: pre-scaling)
    z_scaled: dict[int, torch.Tensor] = field(default_factory=dict)  # B layer -> z after scaling
    attn_out_orig: dict[int, torch.Tensor] = field(default_factory=dict)  # A layer -> attn_out pre-deletion
    q_base: dict[int, torch.Tensor] = field(default_factory=dict)  # B layer -> q pre-modulation
    pattern: dict[int, torch.Tensor] = field(default_factory=dict)  # B layer -> post-softmax pattern
    v_cache: dict[int, torch.Tensor] = field(default_factory=dict)  # B layer -> v (analysis)
    s_B: Optional[torch.Tensor] = None  # [B, nA*d_model] current-run private signal
    o_A: Optional[torch.Tensor] = None  # [B, nA, d_model] per-A-head contributions at p
    delta_q: dict[str, torch.Tensor] = field(default_factory=dict)  # head key -> [B, d_head]
    gamma: dict[str, torch.Tensor] = field(default_factory=dict)  # head key -> [B, d_head]
    hook_calls: dict[str, int] = field(default_factory=dict)
    # set by the backend at build time:
    q_active: bool = False
    v_active: bool = False
    intervention: Optional["Intervention"] = None
    force_signal: bool = False

    def count(self, name: str) -> None:
        self.hook_calls[name] = self.hook_calls.get(name, 0) + 1


# -------------------------------------------------------------- intervention

@dataclass
class Intervention:
    """Causal-test intervention context (spec 10 / 18).

    - off: modulation OUTPUTS are zeroed (delta_q = 0, gamma = 0), which must
      reproduce `delete` exactly; this is distinct from zeroing the adapter
      input (the adapter bias would then still apply).
    - mean_input: sample-independent mean signal (training-set estimate),
      detached; adapter bias retained.
    - shuffled_input: donor rows permuted within the batch (fixed
      derangement); a donor's full A signal group moves together; donor ids
      recorded.
    - paired_swap / swap_*: externally collected, detached donor signal; only
      the private channel input is replaced (external-donor diagnostic).
    """

    name: str = "normal"
    donor_signal: Optional[torch.Tensor] = None  # [B, nA*d_model], detached
    donor_ids: Optional[list] = None
    mean_signal: Optional[torch.Tensor] = None
    shuffle_perm: Optional[torch.Tensor] = None
    q_on: bool = True
    v_on: bool = True

    def __post_init__(self) -> None:
        if self.name not in INTERVENTION_MODES:
            raise ValueError(f"unknown intervention mode: {self.name}")


# ---------------------------------------------------------------- backend

class HookBackend:
    """Builds per-forward hook lists for one architecture mode."""

    def __init__(
        self,
        model: nn.Module,
        circuit: CircuitSpec,
        mode: ArchitectureMode,
        adapters: Optional[nn.Module] = None,
    ) -> None:
        self.model = model
        self.circuit = circuit
        self.mode = mode
        self.adapters = adapters
        circuit.validate()
        self.max_a_layer = max(a.layer for a in circuit.a_heads)
        self.q_branch = None
        self.v_branch = None
        if adapters is not None:
            if mode == "private_q":
                self.q_branch = adapters  # QBranch
            elif mode == "private_v":
                self.v_branch = adapters  # VBranch
            elif mode == "private_qv":
                self.q_branch = getattr(adapters, "q", None)
                self.v_branch = getattr(adapters, "v", None)
            elif mode == "constant_q":
                self.q_branch = adapters  # ConstantQ
            elif mode == "constant_v":
                self.v_branch = adapters  # ConstantV
            elif mode == "constant_qv":
                self.q_branch = getattr(adapters, "q", None)
                self.v_branch = getattr(adapters, "v", None)
        self.pub_adapter = adapters if mode == "public_adapter" else None
        self.q_active = mode in ("private_q", "private_qv", "constant_q", "constant_qv")
        self.v_active = mode in ("private_v", "private_qv", "constant_v", "constant_qv")
        self.is_constant = mode in ("constant_q", "constant_v", "constant_qv")

    # -- signal helpers -----------------------------------------------------

    def ensure_signal(self, state: ForwardState) -> None:
        """Compute s_B = concat(o_A1[p], o_A2[p], ...) from the CURRENT run's
        cached z at A layers. For the first version (max(A.layer) <
        min(B.layer)) all A heads are strictly earlier than every B head.
        Kept in-graph during training (no detach)."""
        if state.s_B is not None:
            return
        o_A: list[torch.Tensor] = []
        for a in self.circuit.signal_order:
            z_l = state.z_base.get(a.layer)
            if z_l is None:
                raise RuntimeError(
                    f"signal computation: z for A layer {a.layer} not cached; "
                    "hook ordering broken"
                )
            W_O = self.model.blocks[a.layer].attn.W_O  # [H, d_head, d_model]
            o = torch.einsum("bd,do->bo", z_l[:, self.p, a.head, :], W_O[a.head])
            o_A.append(o)
        state.o_A = torch.stack(o_A, dim=1)  # [B, nA, d_model]
        state.s_B = torch.cat(o_A, dim=1)  # [B, nA*d_model]

    def _resolve(self, state: ForwardState, intervention: Intervention, which: str):
        """Resolve the signal tensor for branch `which` at hook time.
        Returns None when the branch must output zero (off / branch-off) or
        when no signal applies (constant modes)."""
        if self.is_constant:
            return None
        name = intervention.name
        if name == "off":
            return None
        if name == "q_on_v_off":
            if which == "q":
                self.ensure_signal(state)
                return state.s_B
            return None
        if name == "q_off_v_on":
            if which == "v":
                self.ensure_signal(state)
                return state.s_B
            return None
        if name in ("paired_swap", "swap_both"):
            return intervention.donor_signal
        if name == "swap_q_only":
            if which == "q":
                return intervention.donor_signal
            self.ensure_signal(state)
            return state.s_B
        if name == "swap_v_only":
            if which == "v":
                return intervention.donor_signal
            self.ensure_signal(state)
            return state.s_B
        if name == "mean_input":
            return intervention.mean_signal
        if name == "shuffled_input":
            perm = intervention.shuffle_perm
            assert perm is not None
            self.ensure_signal(state)
            return state.s_B.detach()[perm]
        # normal
        self.ensure_signal(state)
        return state.s_B

    # -- per-layer hook factories (closures over state and p) ---------------

    def _z_hook(self, layer: int, state: ForwardState, p: int, is_a_layer: bool):
        def hook(z: torch.Tensor, hook=None) -> torch.Tensor:
            state.count(f"z_L{layer}")
            state.z_base[layer] = z
            if is_a_layer:
                return z
            if not state.v_active:
                return z
            if state.intervention.name == "off":
                return z
            # (1+gamma) odot z at position p for B heads. The scaled slice is
            # computed OUT OF PLACE from the unmodified z and assembled into
            # a fresh clone: in-place multiplication would corrupt the
            # autograd graph (MulBackward must re-read its input values).
            gamma_scale = torch.ones(
                (z.shape[0], z.shape[2], z.shape[3]),
                device=z.device, dtype=z.dtype,
            )  # [B, H, d_head]
            for h in self.circuit.b_heads_at(layer):
                if self.is_constant:
                    gamma = self.v_branch.value(h)  # [d_head]
                    gamma = gamma.unsqueeze(0).expand(z.shape[0], -1)
                else:
                    s_v = self._resolve(state, state.intervention, "v")
                    if s_v is None:
                        continue
                    gamma = self.v_branch.gamma(h, s_v)  # [B, d_head]
                state.gamma[f"L{h.layer}H{h.head}"] = gamma
                gamma_scale[:, h.head, :] = 1.0 + gamma
            scaled = z[:, p, :, :] * gamma_scale  # out-of-place, reads z only
            out = z.clone()  # fresh storage
            out[:, p, :, :] = scaled  # write into the fresh clone
            state.z_scaled[layer] = out
            return out

        return hook

    def _q_hook(self, layer: int, state: ForwardState, p: int):
        def hook(q: torch.Tensor, hook=None) -> torch.Tensor:
            state.count(f"q_L{layer}")
            state.q_base[layer] = q
            if state.q_active:
                if self.is_constant:
                    if state.intervention.name != "off":
                        q = q.clone()
                        for h in self.circuit.b_heads_at(layer):
                            delta = self.q_branch.value(h)  # [d_head]
                            state.delta_q[f"L{h.layer}H{h.head}"] = (
                                delta.unsqueeze(0).expand(q.shape[0], -1)
                            )
                            q[:, p, h.head, :] = q[:, p, h.head, :] + delta
                else:
                    s_q = self._resolve(state, state.intervention, "q")
                    if s_q is not None:
                        q = q.clone()
                        for h in self.circuit.b_heads_at(layer):
                            delta = self.q_branch.delta(h, s_q)  # [B, d_head]
                            state.delta_q[f"L{h.layer}H{h.head}"] = delta
                            q[:, p, h.head, :] = q[:, p, h.head, :] + delta
            return q

        return hook

    def _pattern_hook(self, layer: int, state: ForwardState):
        def hook(pattern: torch.Tensor, hook=None) -> torch.Tensor:
            state.count(f"pattern_L{layer}")
            state.pattern[layer] = pattern
            return pattern

        return hook

    def _v_hook(self, layer: int, state: ForwardState):
        def hook(v: torch.Tensor, hook=None) -> torch.Tensor:
            state.count(f"v_L{layer}")
            state.v_cache[layer] = v
            return v

        return hook

    def _attn_out_hook(self, layer: int, state: ForwardState, p: int):
        def hook(attn_out: torch.Tensor, hook=None) -> torch.Tensor:
            state.count(f"attn_out_L{layer}")
            state.attn_out_orig[layer] = attn_out
            out = attn_out
            heads = self.circuit.a_heads_at(layer)
            if heads:
                z_l = state.z_base.get(layer)
                if z_l is None:
                    raise RuntimeError("deletion: z not cached before attn_out")
                W_O = self.model.blocks[layer].attn.W_O
                o_sum = None
                for h in heads:
                    o = torch.einsum("bd,do->bo", z_l[:, p, h.head, :], W_O[h.head])
                    o_sum = o if o_sum is None else o_sum + o
                out = attn_out.clone()
                out[:, p, :] = out[:, p, :] - o_sum
            if self.mode == "public_adapter" and layer == self.max_a_layer:
                # All A heads have been computed by this point in the run
                # (layers <= max_a_layer), so the full A cache is available.
                self.ensure_signal(state)
                out = out.clone()
                out[:, p, :] = out[:, p, :] + self.pub_adapter(state.s_B)
            return out

        return hook

    # -- main ----------------------------------------------------------------

    def build_hooks(
        self,
        p: int,
        intervention: Intervention,
        need_analysis: bool = False,
        force_signal: bool = False,
    ) -> tuple[list, ForwardState]:
        """Return (fwd_hooks, state). `p` is the target query position index
        (Q1: final input position, identical for all samples in an
        equal-length bucket)."""
        self.p = p
        state = ForwardState()
        state.q_active = self.q_active
        state.v_active = self.v_active
        state.intervention = intervention
        state.force_signal = force_signal

        hooks: list = []
        for layer in self.circuit.a_layers():
            hooks.append(
                (f"blocks.{layer}.attn.hook_z", self._z_hook(layer, state, p, True))
            )
            hooks.append(
                (f"blocks.{layer}.hook_attn_out", self._attn_out_hook(layer, state, p))
            )
        for layer in self.circuit.b_layers():
            if self.q_active or need_analysis:
                # q hook is also installed for analysis captures (e.g.
                # delete-mode B-query change diagnostics)
                hooks.append(
                    (f"blocks.{layer}.attn.hook_q", self._q_hook(layer, state, p))
                )
            if self.v_active or need_analysis:
                hooks.append(
                    (f"blocks.{layer}.attn.hook_z", self._z_hook(layer, state, p, False))
                )
            if need_analysis:
                hooks.append(
                    (
                        f"blocks.{layer}.attn.hook_pattern",
                        self._pattern_hook(layer, state),
                    )
                )
                hooks.append(
                    (f"blocks.{layer}.attn.hook_v", self._v_hook(layer, state))
                )
        return hooks, state

    # ---------------------------------------------------------------------

    def run(
        self,
        tokens: torch.Tensor,
        p: int,
        intervention: Optional[Intervention] = None,
        attention_mask: Optional[torch.Tensor] = None,
        need_analysis: bool = False,
        force_signal: bool = False,
    ):
        """Run one hooked forward; returns (logits, state)."""
        intervention = intervention or Intervention()
        self._validate_intervention(intervention)
        hooks, state = self.build_hooks(
            p, intervention, need_analysis=need_analysis, force_signal=force_signal
        )
        logits = self.model.run_with_hooks(
            tokens,
            fwd_hooks=hooks,
            attention_mask=attention_mask,
            reset_hooks_end=True,
        )
        if (force_signal or need_analysis) and state.s_B is None:
            if all(a.layer in state.z_base for a in self.circuit.signal_order):
                self.ensure_signal(state)
        return logits, state

    def _validate_intervention(self, intervention: Intervention) -> None:
        if intervention.name == "normal":
            return
        if self.mode in ("original", "delete"):
            raise ValueError(
                f"intervention '{intervention.name}' is not defined for mode {self.mode}"
            )
        if self.is_constant and intervention.name not in ("normal", "off"):
            raise ValueError(
                f"intervention '{intervention.name}' requires a sample-dependent "
                f"signal; mode {self.mode} has none"
            )
        if intervention.name == "off":
            return
        if intervention.name == "mean_input" and intervention.mean_signal is None:
            raise ValueError("mean_input requires mean_signal")
        if intervention.name in (
            "paired_swap", "swap_both", "swap_q_only", "swap_v_only",
        ) and intervention.donor_signal is None:
            raise ValueError(f"{intervention.name} requires donor_signal")
        if intervention.name == "shuffled_input" and intervention.shuffle_perm is None:
            raise ValueError("shuffled_input requires shuffle_perm")

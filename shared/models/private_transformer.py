"""Unified private-model wrapper (spec 2.6 / 15.8).

One forward pass performs: original model forward, A signal caching, A public
writeback deletion, B Query/Value private modulation, and final logits — via
hooks installed inside this wrapper (never scattered in question scripts).
No extra clean forward is used for normal prediction; external donor forwards
are only used by explicitly-labelled causal swap diagnostics.

Modes: original, delete, private_q, private_v, private_qv, public_adapter,
constant_q, constant_v, constant_qv. The three private architectures are
composed from shared components; there is no triplicated forward logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from shared.models.adapters import (
    ConstantQ,
    ConstantV,
    ConstantQV,
    DualBranch,
    PublicAdapter,
    QBranch,
    VBranch,
)
from shared.models.hook_backend import ForwardState, HookBackend, Intervention
from shared.models.routing import (
    ArchitectureMode,
    BudgetRanks,
    CircuitSpec,
    param_counts_for_mode,
)


def build_adapters(
    mode: ArchitectureMode,
    circuit: CircuitSpec,
    ranks: BudgetRanks,
    in_dim: int,
    d_head: int,
    d_model: int,
    seed: Optional[int] = None,
) -> Optional[nn.Module]:
    if mode == "original" or mode == "delete":
        return None
    if mode == "private_q":
        assert ranks.q_rank is not None
        return QBranch(circuit.b_heads, in_dim, d_head, ranks.q_rank, seed=seed)
    if mode == "private_v":
        assert ranks.v_rank is not None
        return VBranch(circuit.b_heads, in_dim, d_head, ranks.v_rank, seed=seed)
    if mode == "private_qv":
        assert ranks.q_rank is not None and ranks.v_rank is not None
        return DualBranch(
            circuit.b_heads, in_dim, d_head, ranks.q_rank, ranks.v_rank, seed=seed
        )
    if mode == "constant_q":
        return ConstantQ(circuit.b_heads, d_head)
    if mode == "constant_v":
        return ConstantV(circuit.b_heads, d_head)
    if mode == "constant_qv":
        return ConstantQV(circuit.b_heads, d_head)
    if mode == "public_adapter":
        assert ranks.pub_rank is not None
        return PublicAdapter(in_dim, d_model, ranks.pub_rank, seed=seed)
    raise ValueError(f"build_adapters: unsupported mode {mode}")


@dataclass
class AnalysisBundle:
    """Per-forward diagnostics (spec 19)."""

    s_B: Optional[torch.Tensor] = None  # [B, nA*d_model]
    o_A: Optional[torch.Tensor] = None  # [B, nA, d_model]
    delta_q: dict[str, torch.Tensor] = field(default_factory=dict)
    gamma: dict[str, torch.Tensor] = field(default_factory=dict)
    q_base: dict[int, torch.Tensor] = field(default_factory=dict)
    z_base: dict[int, torch.Tensor] = field(default_factory=dict)
    z_scaled: dict[int, torch.Tensor] = field(default_factory=dict)
    pattern: dict[int, torch.Tensor] = field(default_factory=dict)
    v_cache: dict[int, torch.Tensor] = field(default_factory=dict)
    attn_out_orig: dict[int, torch.Tensor] = field(default_factory=dict)
    hook_calls: dict[str, int] = field(default_factory=dict)


class PrivateTransformer(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        circuit: CircuitSpec,
        mode: ArchitectureMode,
        ranks: BudgetRanks,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if mode not in ("original", "delete", "private_q", "private_v", "private_qv",
                        "public_adapter", "constant_q", "constant_v", "constant_qv"):
            raise ValueError(f"unknown architecture mode: {mode}")
        circuit.validate()
        self.backbone = backbone
        self.circuit = circuit
        self.mode = mode
        self.ranks = ranks
        self.d_model = backbone.cfg.d_model
        self.d_head = backbone.cfg.d_head
        self.in_dim = circuit.n_a * self.d_model
        self.adapters = build_adapters(
            mode, circuit, ranks, self.in_dim, self.d_head, self.d_model, seed=seed
        )
        # Adapters run on the same device as the backbone.
        dev = next(backbone.parameters()).device
        if self.adapters is not None:
            self.adapters = self.adapters.to(dev)
        self.backend = HookBackend(backbone, circuit, mode, self.adapters)

    # ------------------------------------------------------------ params

    def trainable_parameters(self):
        if self.adapters is None:
            return iter(())
        return iter(self.adapters.parameters())

    def param_stats(self) -> dict:
        return param_counts_for_mode(
            self.mode, self.ranks, self.in_dim, self.d_head, self.d_model, self.circuit.n_b
        )

    def verify_zero_init_equals_delete(self, tokens: torch.Tensor,
                                       attention_mask: Optional[torch.Tensor] = None) -> bool:
        """Zero-initialized private/constant modes must match `delete`
        exactly (same computation graph shape, numerically equal logits)."""
        from shared.models.private_transformer import PrivateTransformer

        if self.mode in ("original", "delete"):
            return True
        delete_model = PrivateTransformer(
            self.backbone, self.circuit, "delete", BudgetRanks()
        )
        with torch.no_grad():
            logits_priv, _ = self.forward(
                tokens, attention_mask=attention_mask, need_analysis=False
            )
            logits_del, _ = delete_model.forward(
                tokens, attention_mask=attention_mask, need_analysis=False
            )
        return bool(torch.allclose(logits_priv, logits_del, atol=1e-6, rtol=1e-6))

    # ----------------------------------------------------------- forward

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        intervention: Optional[Intervention] = None,
        need_analysis: bool = False,
        force_signal: bool = False,
        target_position: Optional[int] = None,
    ):
        """Run one hooked forward. Returns (logits, AnalysisBundle)."""
        if self.mode == "original":
            logits = self.backbone(tokens, attention_mask=attention_mask)
            return logits, AnalysisBundle()
        p = (
            tokens.shape[1] - 1
            if target_position is None
            else target_position
        )
        logits, state = self.backend.run(
            tokens,
            p=p,
            intervention=intervention,
            attention_mask=attention_mask,
            need_analysis=need_analysis,
            force_signal=force_signal,
        )
        return logits, _bundle(state)

    # ------------------------------------------------------------ helpers

    @torch.no_grad()
    def estimate_mean_signal(
        self, tokens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Mean A signal over samples, estimated from NORMAL modified runs
        (architecture active, deletion in effect), no_grad + detached
        (spec 10.3). Returns [1, nA*d_model] broadcastable mean."""
        _, state = self.forward(
            tokens, attention_mask=attention_mask,
            intervention=Intervention(name="normal"), force_signal=True,
        )
        assert state.s_B is not None
        return state.s_B.detach().mean(dim=0, keepdim=True)

    @torch.no_grad()
    def collect_donor_signal(
        self, tokens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """External donor run: detached s_B of the normal private forward
        (spec 10.5: donor signals come from independent normal runs and are
        detached; only the private channel input is replaced in recipients)."""
        _, state = self.forward(
            tokens, attention_mask=attention_mask,
            intervention=Intervention(name="normal"), force_signal=True,
        )
        assert state.s_B is not None
        return state.s_B.detach()


def _bundle(state: ForwardState) -> AnalysisBundle:
    return AnalysisBundle(
        s_B=state.s_B,
        o_A=state.o_A,
        delta_q=state.delta_q,
        gamma=state.gamma,
        q_base=state.q_base,
        z_base=state.z_base,
        z_scaled=state.z_scaled,
        pattern=state.pattern,
        v_cache=state.v_cache,
        attn_out_orig=state.attn_out_orig,
        hook_calls=state.hook_calls,
    )

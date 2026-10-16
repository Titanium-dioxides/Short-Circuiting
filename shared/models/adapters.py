"""Trainable modulation modules.

- ``LowRankAffine``: g(s) = W_up (W_down s) + b. W_down is randomly
  initialized (normal), W_up and b are zero, so the initial modulation is
  exactly zero (spec 2.4 / 15.3). No nonlinearity, LayerNorm or extra
  writeback in the first version.
- ``QBranch`` / ``VBranch``: per-receiving-head-B g_Q,B / g_V,B modules.
- ``DualBranch``: independent Q and V branches (no shared hidden layer, no
  cross-branch interaction module; a shared bottleneck is a future variant
  only, spec 15.6).
- ``ConstantQ/V/QV``: trainable, sample-independent per-head delta_q / gamma
  (static-compensation controls, spec 16). Zero-initialized; no useless
  padding parameters are added to fake parameter equality.
- ``PublicAdapter``: writes adapter(s_B) back into the public residual at the
  last A layer's attention residual-add point (spec 8). Sized to the
  corresponding private model's actual budget by the experiment layer.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from shared.models.routing import HeadRef

INIT_STD = 0.02  # normal-init std for W_down ("正常随机初始化")


class LowRankAffine(nn.Module):
    """g(s) = W_up (W_down s) + b, zero-initialized output."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        gen = None
        if seed is not None:
            gen = torch.Generator().manual_seed(seed)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        W_down = torch.empty(rank, in_dim)
        nn.init.normal_(W_down, std=INIT_STD, generator=gen)
        self.W_down = nn.Parameter(W_down)
        self.W_up = nn.Parameter(torch.zeros(out_dim, rank))
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # s: [..., in_dim] -> [..., out_dim]
        h = torch.einsum("...i,ri->...r", s, self.W_down)
        return torch.einsum("...r,or->...o", h, self.W_up) + self.b

    def zero_output_guaranteed(self) -> bool:
        return bool(self.W_up.abs().max().item() == 0.0 and self.b.abs().max().item() == 0.0)


class _HeadModuleDict(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mods: nn.ModuleDict = nn.ModuleDict()

    def _key(self, head: HeadRef) -> str:
        return f"L{head.layer}H{head.head}"

    def __getitem__(self, head: HeadRef) -> nn.Module:
        return self.mods[self._key(head)]

    def __contains__(self, head: HeadRef) -> bool:
        return self._key(head) in self.mods


class QBranch(_HeadModuleDict):
    """Per-B-head Q modulation: delta_q = g_Q,B(s_B), added to the base query."""

    def __init__(
        self,
        b_heads: tuple[HeadRef, ...],
        in_dim: int,
        d_head: int,
        rank: int,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.b_heads = tuple(sorted(b_heads))
        self.in_dim = in_dim
        self.out_dim = d_head
        self.rank = rank
        for i, h in enumerate(self.b_heads):
            s = None if seed is None else seed + 1000 * i
            self.mods[self._key(h)] = LowRankAffine(in_dim, d_head, rank, seed=s)

    def delta(self, head: HeadRef, s_B: torch.Tensor) -> torch.Tensor:
        """s_B: [B, in_dim] -> [B, d_head]."""
        return self[head](s_B)


class VBranch(_HeadModuleDict):
    """Per-B-head V feature scaling: gamma = g_V,B(s_B); z' = (1+gamma) * z
    applied at hook_z (post-aggregation, pre-W_O), the exact algebraic
    equivalent of scaling v for the target query (spec 15.4/15.5)."""

    def __init__(
        self,
        b_heads: tuple[HeadRef, ...],
        in_dim: int,
        d_head: int,
        rank: int,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.b_heads = tuple(sorted(b_heads))
        self.in_dim = in_dim
        self.out_dim = d_head
        self.rank = rank
        for i, h in enumerate(self.b_heads):
            s = None if seed is None else seed + 2000 * i
            self.mods[self._key(h)] = LowRankAffine(in_dim, d_head, rank, seed=s)

    def gamma(self, head: HeadRef, s_B: torch.Tensor) -> torch.Tensor:
        return self[head](s_B)


class DualBranch(nn.Module):
    """Independent Q and V branches (spec 15.6)."""

    def __init__(
        self,
        b_heads: tuple[HeadRef, ...],
        in_dim: int,
        d_head: int,
        q_rank: int,
        v_rank: int,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.q = QBranch(b_heads, in_dim, d_head, q_rank, seed=seed)
        self.v = VBranch(b_heads, in_dim, d_head, v_rank, seed=seed)


class ConstantHeads(nn.Module):
    """Per-B-head trainable, sample-independent vector (delta_q or gamma)."""

    def __init__(self, b_heads: tuple[HeadRef, ...], d_head: int) -> None:
        super().__init__()
        self.b_heads = tuple(sorted(b_heads))
        self.d_head = d_head
        self.mods = nn.ParameterDict(
            {self._key(h): nn.Parameter(torch.zeros(d_head)) for h in self.b_heads}
        )

    @staticmethod
    def _key(head: HeadRef) -> str:
        return f"L{head.layer}H{head.head}"

    def value(self, head: HeadRef) -> torch.Tensor:
        return self.mods[self._key(head)]


class ConstantQ(ConstantHeads):
    """Static delta_q per B head; never reads the A signal (spec 16)."""


class ConstantV(ConstantHeads):
    """Static gamma per B head, applied with exactly the same feature-scaling
    location and form as private_v (spec 16)."""


class ConstantQV(nn.Module):
    def __init__(self, b_heads: tuple[HeadRef, ...], d_head: int) -> None:
        super().__init__()
        self.q = ConstantQ(b_heads, d_head)
        self.v = ConstantV(b_heads, d_head)


class PublicAdapter(nn.Module):
    """Writes adapter(s_B) into the public residual stream at the last A
    layer's attention residual-add point (after A writeback deletion),
    before all B layers (spec 8)."""

    def __init__(self, in_dim: int, d_model: int, rank: int, seed: Optional[int] = None) -> None:
        super().__init__()
        self.core = LowRankAffine(in_dim, d_model, rank, seed=seed)

    def forward(self, s_B: torch.Tensor) -> torch.Tensor:
        return self.core(s_B)

"""Generic activation patching for calibration diagnostics (spec 7).

Used by Q1 calibration for B-query patching on paired samples and for the
query-repair diagnostic. These are diagnostics over hooks; they are NOT the
architecture itself and never replace the in-run private channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ActivationPatch:
    """A hook-level patch: set rows `batch_idx` at position `pos` (and heads
    `heads`, when the hook tensor is per-head) to `value`."""

    hook_name: str
    value: torch.Tensor  # [n_rows, ...] matching the patched slice
    batch_idx: list[int] = field(default_factory=list)
    pos: Optional[int] = None
    heads: Optional[list[int]] = None
    mode: str = "replace"  # replace | add

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor
        if self.batch_idx:
            idx = torch.tensor(self.batch_idx, device=tensor.device)
            if self.pos is not None and self.heads is not None:
                # [B, pos, H, d] hook (q/k/v/z)
                sub = out[idx, self.pos, :, :]
                if self.mode == "replace":
                    sub = sub.clone()
                    sub[:, self.heads, :] = self.value.to(sub.dtype)
                else:
                    sub = sub.clone()
                    sub[:, self.heads, :] = sub[:, self.heads, :] + self.value.to(sub.dtype)
                out = out.clone()
                out[idx, self.pos, :, :] = sub
            elif self.pos is not None:
                # [B, pos, d] hook
                if self.mode == "replace":
                    out = out.clone()
                    out[idx, self.pos, :] = self.value.to(out.dtype)
                else:
                    out = out.clone()
                    out[idx, self.pos, :] = out[idx, self.pos, :] + self.value.to(out.dtype)
            else:
                raise ValueError("ActivationPatch requires pos for per-position patching")
        return out


def patch_forward(
    model,
    tokens: torch.Tensor,
    patches: list[ActivationPatch],
    attention_mask: Optional[torch.Tensor] = None,
):
    """Run one forward with a list of activation patches installed as
    temporary fwd_hooks. Returns logits."""
    fwd_hooks = [
        (p.hook_name, (lambda t, hook=None, p=p: p.apply(t))) for p in patches
    ]
    return model.run_with_hooks(
        tokens, fwd_hooks=fwd_hooks, attention_mask=attention_mask, reset_hooks_end=True
    )

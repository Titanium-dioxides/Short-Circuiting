"""Checkpointing and training-integrity checks (spec 9/20).

A checkpoint saves ALL trainable branches plus the architecture mode, so the
saved artifact is sufficient to reconstruct the exact model (spec 20.12).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

from shared.models.backbone import fingerprint_matches, param_fingerprint
from shared.models.private_transformer import PrivateTransformer
from shared.utils.experiment_io import load_json, save_json


def verify_backbone_frozen(model: PrivateTransformer, ref_fingerprint: dict) -> bool:
    """True iff the GPT-2 backbone parameters are unchanged vs the reference
    fingerprint (shape + fp64 sum)."""
    return fingerprint_matches(param_fingerprint(model.backbone), ref_fingerprint)


def verify_gradients(model: PrivateTransformer, tokens: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None,
                     correct_ids: Optional[torch.Tensor] = None) -> dict:
    """One backward pass with a dummy NLL target at the final position;
    reports per-branch grad norms. Raises if any trainable param has no
    gradient while the backbone stays frozen (spec 9: valid gradients)."""
    import torch.nn.functional as F

    logits, _ = model.forward(tokens, attention_mask=attention_mask)
    logp = F.log_softmax(logits[:, -1, :], dim=-1)
    if correct_ids is None:
        correct_ids = logits[:, -1, :].argmax(dim=-1)
    loss = -logp[torch.arange(logp.shape[0]), correct_ids].mean()
    loss.backward()
    report: dict = {"branches": {}, "backbone_has_grad": False}
    if model.adapters is not None:
        for name, p in model.adapters.named_parameters():
            g = p.grad
            report["branches"][name] = (
                0.0 if g is None else float(g.norm().item())
            )
        missing = [n for n, v in report["branches"].items() if v == 0.0]
        if missing:
            raise RuntimeError(f"trainable params without gradient: {missing}")
    for p in model.backbone.parameters():
        if p.grad is not None:
            report["backbone_has_grad"] = True
            break
    model.zero_grad(set_to_none=True)
    return report


def save_private_checkpoint(
    model: PrivateTransformer,
    path: str | Path,
    meta: Optional[dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "architecture_mode": model.mode,
        "ranks": model.ranks.to_dict(),
        "circuit": model.circuit.to_dict(),
        "meta": meta or {},
    }
    torch.save(payload, path.with_suffix(".meta.pt"))
    state = (
        model.adapters.state_dict()
        if model.adapters is not None
        else {}
    )
    torch.save(state, path)


def load_private_checkpoint(
    model: PrivateTransformer,
    path: str | Path,
    ref_fingerprint: Optional[dict] = None,
) -> dict:
    """Load branch weights into `model` (which must have been built with the
    same architecture mode). Verifies: mode match, backbone unchanged.
    Returns the meta payload."""
    path = Path(path)
    meta_payload = torch.load(path.with_suffix(".meta.pt"), map_location="cpu",
                              weights_only=False)
    if meta_payload["architecture_mode"] != model.mode:
        raise ValueError(
            f"checkpoint mode {meta_payload['architecture_mode']} != model mode {model.mode}"
        )
    state = torch.load(path, map_location="cpu", weights_only=False)
    if model.adapters is not None:
        model.adapters.load_state_dict(state)
    elif state:
        raise ValueError("checkpoint has branch weights but model has no adapters")
    if ref_fingerprint is not None:
        if not verify_backbone_frozen(model, ref_fingerprint):
            raise RuntimeError("backbone parameters changed vs reference fingerprint")
    return meta_payload

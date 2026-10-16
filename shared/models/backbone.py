"""Backbone: GPT-2 small loading (frozen, eval) via TransformerLens.

This machine has no internet access; the official `openai-community/gpt2`
weights are present in the local HuggingFace cache. Loading is done from the
cached snapshot: the HF model is loaded with `local_files_only` and handed to
`HookedTransformer.from_pretrained(..., hf_model=hf)`.

Parameter transforms are disabled (`fold_ln=False, center_writing_weights=False,
fold_value_biases=False`) so the original parameterization is kept — in
particular `b_O` remains a separate parameter, which the deletion definition
(spec 2.3: attn_out = sum_h o_h + b_O) relies on. All experiment groups use
exactly the same loading options.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# This machine has no internet access; force offline resolution for all
# HuggingFace/TransformerLens code paths (weights live in the local cache).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn as nn
from transformer_lens import HookedTransformer

GPT2_SMALL_SNAPSHOT = Path(
    os.environ.get(
        "GPT2_SNAPSHOT",
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/"
            "607a30d783dfa663caf39e06633721c8d4cfcd7e"
        ),
    )
)


def load_hf_model(snapshot: Optional[str | Path] = None):
    from transformers import AutoModelForCausalLM

    snap = Path(snapshot) if snapshot else GPT2_SMALL_SNAPSHOT
    if not snap.exists():
        raise FileNotFoundError(
            f"GPT-2 snapshot not found at {snap}; set GPT2_SNAPSHOT env var "
            "to the local snapshot directory"
        )
    return AutoModelForCausalLM.from_pretrained(str(snap), local_files_only=True)


def load_tokenizer(snapshot: Optional[str | Path] = None):
    from transformers import AutoTokenizer

    snap = Path(snapshot) if snapshot else GPT2_SMALL_SNAPSHOT
    return AutoTokenizer.from_pretrained(str(snap), local_files_only=True)


def load_backbone(
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    snapshot: Optional[str | Path] = None,
) -> HookedTransformer:
    """Load GPT-2 small as a HookedTransformer, eval mode, all params frozen.

    Loading options (recorded in environment.json per run):
      fold_ln=False, center_writing_weights=False, fold_value_biases=False
    """
    hf = load_hf_model(snapshot)
    tokenizer = load_tokenizer(snapshot)
    model = HookedTransformer.from_pretrained(
        "gpt2",
        hf_model=hf,
        tokenizer=tokenizer,
        fold_ln=False,
        center_writing_weights=False,
        fold_value_biases=False,
        dtype=dtype,
        device=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


LOAD_OPTIONS = {
    "model": "gpt2 (openai-community/gpt2, GPT-2 small)",
    "snapshot": str(GPT2_SMALL_SNAPSHOT),
    "fold_ln": False,
    "center_writing_weights": False,
    "fold_value_biases": False,
    "note": "no extra parameter transforms; b_O stays a separate parameter",
}


def param_fingerprint(model: nn.Module) -> dict[str, tuple]:
    """Cheap fingerprint of the frozen backbone: per-param name -> (shape,
    float64 sum). Used to verify the backbone never changes during training
    and across checkpoint save/load."""
    fp: dict[str, tuple] = {}
    for name, p in model.named_parameters():
        with torch.no_grad():
            fp[name] = (
                tuple(p.shape),
                float(p.detach().float().double().sum().item()),
            )
    return fp


def fingerprint_matches(fp_a: dict[str, tuple], fp_b: dict[str, tuple]) -> bool:
    if set(fp_a.keys()) != set(fp_b.keys()):
        return False
    for k in fp_a:
        sa, va = fp_a[k]
        sb, vb = fp_b[k]
        if sa != sb or abs(va - vb) > 1e-3:
            return False
    return True

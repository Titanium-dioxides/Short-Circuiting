"""Signal controls: builders for the causal-test interventions (spec 10/18).

All swap interventions are explicitly labelled external-donor diagnostics:
the donor signal is collected by an independent normal private forward and
detached; only the private channel input is replaced in the recipient run.
"""

from __future__ import annotations

from typing import Optional

import torch

from shared.models.hook_backend import Intervention
from shared.models.private_transformer import PrivateTransformer


def off() -> Intervention:
    """Full off: delta_q = 0 and gamma = 0 (identity scaling). The resulting
    forward must equal `delete` exactly."""
    return Intervention(name="off", q_on=False, v_on=False)


def branch_ablation(q_on: bool, v_on: bool) -> Intervention:
    """Inference-time branch ablation on the SAME trained QV checkpoint
    (spec 18: not to be confused with independently trained Q-only/V-only)."""
    name = {
        (True, False): "q_on_v_off",
        (False, True): "q_off_v_on",
        (False, False): "off",
        (True, True): "normal",
    }[(q_on, v_on)]
    return Intervention(name=name, q_on=q_on, v_on=v_on)


def derangement_perm(n: int, shift: int = 1) -> torch.Tensor:
    """Fixed cyclic shift i -> (i + shift) mod n; a derangement for n >= 2."""
    if n < 2:
        raise ValueError("derangement requires batch size >= 2")
    return (torch.arange(n) + shift) % n


def shuffled(perm: torch.Tensor, donor_ids: Optional[list] = None) -> Intervention:
    """shuffled_input: rows permuted within the current batch; each donor's
    full A signal group (the whole s_B vector) moves together."""
    return Intervention(
        name="shuffled_input", shuffle_perm=perm, donor_ids=donor_ids
    )


def mean_input(mean_signal: torch.Tensor) -> Intervention:
    """mean_input: training-set mean signal (detached); adapter bias kept."""
    return Intervention(name="mean_input", mean_signal=mean_signal)


def swap_donor(
    donor_signal: torch.Tensor,
    donor_ids: Optional[list] = None,
    source: str = "both",
) -> Intervention:
    """Donor swap of the private channel input only (external-donor
    diagnostic). source: 'q_only' | 'v_only' | 'both' (spec 18)."""
    name = {"q_only": "swap_q_only", "v_only": "swap_v_only", "both": "swap_both"}[source]
    return Intervention(name=name, donor_signal=donor_signal, donor_ids=donor_ids)


def collect_donor(
    model: PrivateTransformer,
    tokens: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[list]]:
    """External donor collection: independent normal private run, detached
    s_B. Returns (signal, donor_ids=None)."""
    signal = model.collect_donor_signal(tokens, attention_mask=attention_mask)
    return signal, None

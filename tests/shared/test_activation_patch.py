"""Regression: activation-patch helpers must follow TL 3.2.1's hook calling
convention hook(tensor, hook=self)."""

import torch

from shared.interventions.activation_patch import ActivationPatch, patch_forward
from tests.shared.conftest import TOKENS


def test_patch_forward_runs_with_tl_hook_convention(backbone):
    obs = {}

    def q_hook(q, hook=None):
        obs["q"] = q
        return q

    with torch.no_grad():
        backbone.run_with_hooks(
            TOKENS[0:1], fwd_hooks=[("blocks.0.attn.hook_q", q_hook)]
        )
        base_q = obs["q"]  # [1, S, H, dh]
        patch = ActivationPatch(
            hook_name="blocks.0.attn.hook_q",
            value=base_q[:, -1, 0:1, :] * 2.0,
            batch_idx=[0],
            pos=TOKENS.shape[1] - 1,
            heads=[0],
        )
        logits = patch_forward(backbone, TOKENS[0:1], [patch])
        logits_plain = backbone(TOKENS[0:1])
    assert logits.shape == logits_plain.shape
    # patching head 0's query at the last position must change that
    # position's logits
    assert not torch.allclose(logits[:, -1, :], logits_plain[:, -1, :], atol=1e-6)

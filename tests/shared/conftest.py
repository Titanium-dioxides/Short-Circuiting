"""Shared fixtures: one session-scoped frozen GPT-2 backbone (CPU, FP32) and
a small A/B circuit used by all architecture tests."""

import pytest
import torch

from shared.models import (
    BudgetRanks,
    CircuitSpec,
    HeadRef,
    load_backbone,
    PrivateTransformer,
)

TOKENS = torch.tensor(
    [
        [464, 717, 1502, 11, 220, 318, 262, 3139],
        [319, 428, 1588, 11, 464, 717, 262, 3139],
    ]
)  # short token sequences, S=8; p = 7


@pytest.fixture(scope="session")
def backbone():
    model = load_backbone(device="cpu")
    return model


@pytest.fixture(scope="session")
def circuit():
    # A strictly below B (max A layer 8 < min B layer 9); 0-based ids.
    return CircuitSpec(
        a_heads=(HeadRef(7, 3), HeadRef(8, 6)),
        b_heads=(HeadRef(9, 6), HeadRef(10, 0)),
    )


@pytest.fixture()
def tokens():
    return TOKENS.clone()


def make_model(backbone, circuit, mode, ranks=None, seed=0):
    if ranks is None:
        ranks = BudgetRanks(q_rank=4, v_rank=4, pub_rank=4)
    return PrivateTransformer(backbone, circuit, mode, ranks, seed=seed)

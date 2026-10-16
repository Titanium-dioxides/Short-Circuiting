"""Q1 test fixtures: reuse the session-scoped backbone/circuit from the
shared suite; tiny deterministic datasets are generated per test."""

import pytest

from tests.shared.conftest import backbone, circuit, tokens, make_model  # noqa: F401

from shared.models import load_tokenizer
from questions.q1_private_communication.src.data import generate_ioi

TINY_POOLS = {
    "A": ("Mary", "John", "Alice", "Bob"),
    "B": ("Linda", "Mark", "Emma", "David"),
    "C": ("Maria", "Peter", "Susan", "Henry"),
    "D": ("Carol", "Eric", "Helen", "Kevin"),
}


@pytest.fixture(scope="session")
def tiny_dataset():
    tok = load_tokenizer()
    return generate_ioi(
        tok,
        seed=0,
        n_train_templates=1,
        n_holdout_templates=1,
        name_pools=TINY_POOLS,
        forms=("name",),
    )

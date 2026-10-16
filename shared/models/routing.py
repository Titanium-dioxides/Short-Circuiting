"""Routing: A/B circuit specification, modes, budget regimes and validation.

Field separation required by the spec (section 15.8): ``architecture_mode``,
``intervention_mode`` and ``budget_regime`` are distinct fields and are never
merged into one ambiguous string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

# ---------------------------------------------------------------- head refs

@dataclass(frozen=True, order=True)
class HeadRef:
    """0-based (layer, head) reference. Human-readable form uses 0-based ids
    and always spells out the convention: "L7H3 (0-based)"."""

    layer: int
    head: int

    def __str__(self) -> str:
        return f"L{self.layer}H{self.head}"

    @classmethod
    def from_spec(cls, layer: int, head: int, source_indexing: Literal[0, 1] = 0) -> "HeadRef":
        """Convert a literature head id into a 0-based HeadRef.

        Literature ids are often 1-based (layer 9 = 9th layer). The caller
        must state which convention the source uses; the stored ref is always
        0-based. This keeps the indexing question explicit instead of silent.
        """
        layer0 = layer - 1 if source_indexing == 1 else layer
        return cls(layer=layer0, head=head)


# ------------------------------------------------------------------- modes

ARCHITECTURE_MODES = (
    "original",
    "delete",
    "private_q",
    "private_v",
    "private_qv",
    "public_adapter",
    "constant_q",
    "constant_v",
    "constant_qv",
)
ArchitectureMode = Literal[
    "original",
    "delete",
    "private_q",
    "private_v",
    "private_qv",
    "public_adapter",
    "constant_q",
    "constant_v",
    "constant_qv",
]

PRIVATE_MODE = Literal["q_only", "v_only", "qv"]

# private_mode is the spec 15.8 config key for private architectures.
PRIVATE_MODE_OF_ARCH: dict[str, Optional[str]] = {
    "private_q": "q_only",
    "private_v": "v_only",
    "private_qv": "qv",
    "original": None,
    "delete": None,
    "public_adapter": None,
    "constant_q": None,
    "constant_v": None,
    "constant_qv": None,
}

INTERVENTION_MODES = (
    "normal",
    "off",
    "mean_input",
    "shuffled_input",
    "paired_swap",
    # Q+V branch ablations (inference-time, same trained checkpoint):
    "q_on_v_off",
    "q_off_v_on",
    # Q+V selective donor swaps (all marked external-donor diagnostics):
    "swap_q_only",
    "swap_v_only",
    "swap_both",
)
InterventionMode = Literal[
    "normal",
    "off",
    "mean_input",
    "shuffled_input",
    "paired_swap",
    "q_on_v_off",
    "q_off_v_on",
    "swap_q_only",
    "swap_v_only",
    "swap_both",
]

BUDGET_REGIMES = ("fixed_branch_capacity", "fixed_total_budget")
BudgetRegime = Literal["fixed_branch_capacity", "fixed_total_budget"]


# --------------------------------------------------------------- circuit

@dataclass
class CircuitSpec:
    """A = sender heads (S-inhibition candidates), B = receiver heads
    (name-mover candidates). Every A->B pair must satisfy A.layer < B.layer;
    the first version additionally enforces max(A.layers) < min(B.layers).
    """

    a_heads: tuple[HeadRef, ...]
    b_heads: tuple[HeadRef, ...]
    target_position: str = "last"  # resolution rule for position p (Q1: last input pos)
    source: str = ""
    selection_notes: str = ""

    def validate(self) -> None:
        if not self.a_heads:
            raise ValueError("CircuitSpec: A head set must not be empty")
        if not self.b_heads:
            raise ValueError("CircuitSpec: B head set must not be empty")
        for a in self.a_heads:
            for b in self.b_heads:
                if a.layer >= b.layer:
                    raise ValueError(
                        f"CircuitSpec: illegal connection A={a} -> B={b}: "
                        f"requires A.layer < B.layer"
                    )
        max_a = max(a.layer for a in self.a_heads)
        min_b = min(b.layer for b in self.b_heads)
        if not (max_a < min_b):
            raise ValueError(
                "CircuitSpec: first version requires max(A.layers) < min(B.layers); "
                f"got max(A)={max_a}, min(B)={min_b}"
            )

    def a_layers(self) -> list[int]:
        return sorted({a.layer for a in self.a_heads})

    def b_layers(self) -> list[int]:
        return sorted({b.layer for b in self.b_heads})

    def a_heads_at(self, layer: int) -> tuple[HeadRef, ...]:
        return tuple(a for a in self.a_heads if a.layer == layer)

    def b_heads_at(self, layer: int) -> tuple[HeadRef, ...]:
        return tuple(b for b in self.b_heads if b.layer == layer)

    def a_before(self, layer: int) -> tuple[HeadRef, ...]:
        """A heads strictly earlier than `layer`, in (layer, head) order."""
        return tuple(a for a in sorted(self.a_heads) if a.layer < layer)

    @property
    def signal_order(self) -> tuple[HeadRef, ...]:
        """Canonical concatenation order of s_B."""
        return tuple(sorted(self.a_heads))

    @property
    def n_a(self) -> int:
        return len(self.a_heads)

    @property
    def n_b(self) -> int:
        return len(self.b_heads)

    def to_dict(self) -> dict:
        return {
            "a_heads": [str(a) for a in sorted(self.a_heads)],
            "b_heads": [str(b) for b in sorted(self.b_heads)],
            "target_position": self.target_position,
            "source": self.source,
            "selection_notes": self.selection_notes,
            "indexing": "0-based layer and head ids",
        }


# ----------------------------------------------------------------- budget

@dataclass
class BudgetConfig:
    regime: BudgetRegime
    # fixed_branch_capacity: every single branch uses this rank
    branch_rank: Optional[int] = None
    # fixed_total_budget: approximate total trainable parameter budget
    total_params: Optional[int] = None


@dataclass
class BudgetRanks:
    """Resolved low-rank sizes for one architecture mode.

    q_rank / v_rank: per-B-head LowRankAffine ranks.
    pub_rank: public adapter rank (public_adapter mode only).
    constant groups carry no rank (per-head trainable vectors).
    """

    q_rank: Optional[int] = None
    v_rank: Optional[int] = None
    pub_rank: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "q_rank": self.q_rank,
            "v_rank": self.v_rank,
            "pub_rank": self.pub_rank,
        }


def adapter_param_count(
    in_dim: int, out_dim: int, rank: int, n_heads: int
) -> int:
    """Trainable params of per-head LowRankAffine modules over n_heads heads:
    rank*in_dim (W_down) + out_dim*rank (W_up) + out_dim (b), per head."""
    per_head = rank * in_dim + out_dim * rank + out_dim
    return per_head * n_heads


def max_fitting_rank(
    in_dim: int, out_dim: int, n_heads: int, budget: int, branches: int = 1
) -> int:
    """Largest positive rank whose param count is <= budget for `branches`
    independent branches (budget shared equally by default)."""
    per_branch_budget = budget // branches
    rank = 1
    best = 0
    # params grow linearly in rank; scan up to a generous cap
    for r in range(1, 4096):
        if adapter_param_count(in_dim, out_dim, r, n_heads) * branches > budget:
            break
        if adapter_param_count(in_dim, out_dim, r, n_heads) <= per_branch_budget:
            best = r
        rank = r
    return best


def resolve_budget(
    budget: BudgetConfig,
    mode: ArchitectureMode,
    in_dim: int,
    d_head: int,
    d_model: int,
    n_b: int,
    ref_total: Optional[int] = None,
) -> BudgetRanks:
    """Resolve ranks for one architecture mode under the given regime.

    - fixed_branch_capacity: Q-only/V-only use rank r; Q+V uses r for BOTH
      branches (more params than single-branch modes; used to observe the
      effect of adding a second modulation capability).
    - fixed_total_budget: q_only/v_only get the largest single-branch rank
      that fits P; qv splits P equally between the two branches (default),
      using the largest rank that fits per branch. No useless params are
      added to pad budgets. public_adapter matches `ref_total` (the
      corresponding private model's actual trainable param count).
    """
    ranks = BudgetRanks()
    if mode in ("original", "delete"):
        return ranks
    if mode in ("constant_q", "constant_v", "constant_qv"):
        return ranks  # per-head vectors, no rank; param counts reported separately

    if mode == "public_adapter":
        target = ref_total if ref_total is not None else budget.total_params
        if target is None:
            raise ValueError("public_adapter needs ref_total or total_params")
        r = 1
        best = 0
        for cand in range(1, 4096):
            if cand * in_dim + d_model * cand + d_model > target:
                break
            best = cand
            r = cand
        if best == 0:
            raise ValueError(
                f"public_adapter: no positive rank fits target {target} "
                f"(in_dim={in_dim}, d_model={d_model})"
            )
        ranks.pub_rank = best
        return ranks

    if budget.regime == "fixed_branch_capacity":
        r = budget.branch_rank
        if r is None or r < 1:
            raise ValueError("fixed_branch_capacity requires branch_rank >= 1")
        if mode == "private_q":
            ranks.q_rank = r
        elif mode == "private_v":
            ranks.v_rank = r
        elif mode == "private_qv":
            ranks.q_rank = r
            ranks.v_rank = r
        return ranks

    if budget.regime == "fixed_total_budget":
        P = budget.total_params
        if P is None or P <= 0:
            raise ValueError("fixed_total_budget requires total_params > 0")
        if mode in ("private_q", "private_v"):
            r = max_fitting_rank(in_dim, d_head, n_b, P, branches=1)
            if r == 0:
                raise ValueError(f"no positive rank fits budget P={P}")
            if mode == "private_q":
                ranks.q_rank = r
            else:
                ranks.v_rank = r
            return ranks
        if mode == "private_qv":
            r = max_fitting_rank(in_dim, d_head, n_b, P, branches=2)
            if r == 0:
                raise ValueError(f"qv: no positive rank fits budget P={P} with 2 branches")
            ranks.q_rank = r
            ranks.v_rank = r
            return ranks

    raise ValueError(f"resolve_budget: unsupported mode {mode}")


def budget_resolution_table(
    budget: BudgetConfig,
    in_dim: int,
    d_head: int,
    d_model: int,
    n_b: int,
) -> dict[str, dict]:
    """Resolve ranks + exact param counts for every trainable architecture
    mode. Records per-branch counts, totals, and (for fixed_total_budget)
    the deviation from P. Also sizes public_adapter to the private_qv total
    (the largest private variant) per spec section 16."""
    table: dict[str, dict] = {}
    priv_totals: dict[str, int] = {}
    for mode in ("private_q", "private_v", "private_qv"):
        ranks = resolve_budget(budget, mode, in_dim, d_head, d_model, n_b)
        counts = param_counts_for_mode(mode, ranks, in_dim, d_head, d_model, n_b)
        priv_totals[mode] = counts["total"]
        table[mode] = {"ranks": ranks.to_dict(), **counts}
    # public_adapter sized to the corresponding private model's actual budget:
    # the largest private variant (qv) by default; per-budget-regime naming
    # is added by the experiment layer.
    ref = priv_totals["private_qv"]
    ranks_pub = resolve_budget(
        budget, "public_adapter", in_dim, d_head, d_model, n_b, ref_total=ref
    )
    pub_counts = param_counts_for_mode(
        "public_adapter", ranks_pub, in_dim, d_head, d_model, n_b
    )
    table["public_adapter"] = {"ranks": ranks_pub.to_dict(), **pub_counts}
    for mode in ("constant_q", "constant_v", "constant_qv"):
        counts = param_counts_for_mode(mode, BudgetRanks(), in_dim, d_head, d_model, n_b)
        table[mode] = {"ranks": {}, **counts}
    if budget.regime == "fixed_total_budget":
        for mode, info in table.items():
            if info.get("total") is not None:
                info["deviation_from_budget"] = info["total"] - budget.total_params
    return table


def param_counts_for_mode(
    mode: ArchitectureMode,
    ranks: BudgetRanks,
    in_dim: int,
    d_head: int,
    d_model: int,
    n_b: int,
) -> dict:
    counts: dict = {"q_branch": 0, "v_branch": 0, "public_adapter": 0, "constants": 0}
    if mode == "private_q" and ranks.q_rank:
        counts["q_branch"] = adapter_param_count(in_dim, d_head, ranks.q_rank, n_b)
    elif mode == "private_v" and ranks.v_rank:
        counts["v_branch"] = adapter_param_count(in_dim, d_head, ranks.v_rank, n_b)
    elif mode == "private_qv":
        if ranks.q_rank:
            counts["q_branch"] = adapter_param_count(in_dim, d_head, ranks.q_rank, n_b)
        if ranks.v_rank:
            counts["v_branch"] = adapter_param_count(in_dim, d_head, ranks.v_rank, n_b)
    elif mode == "public_adapter" and ranks.pub_rank:
        counts["public_adapter"] = (
            ranks.pub_rank * in_dim + d_model * ranks.pub_rank + d_model
        )
    elif mode == "constant_q":
        counts["constants"] = n_b * d_head
    elif mode == "constant_v":
        counts["constants"] = n_b * d_head
    elif mode == "constant_qv":
        counts["constants"] = 2 * n_b * d_head
    counts["total"] = (
        counts["q_branch"] + counts["v_branch"] + counts["public_adapter"] + counts["constants"]
    )
    return counts


def validate_budget_ranks(
    budget: BudgetConfig,
    in_dim: int,
    d_head: int,
    d_model: int,
    n_b: int,
) -> dict[str, dict]:
    """Full budget table + invariants; raises on any violation.

    Invariants (spec 17): positive effective ranks; qv splits the total
    budget equally by default; no padding params; recorded deviations.
    """
    table = budget_resolution_table(budget, in_dim, d_head, d_model, n_b)
    for mode in ("private_q", "private_v", "private_qv"):
        t = table[mode]
        if budget.regime == "fixed_branch_capacity":
            # qv deliberately has more params than single-branch modes
            if mode == "private_qv":
                assert t["total"] > table["private_q"]["total"], (
                    "qv must have more params than q-only under fixed_branch_capacity"
                )
            else:
                assert t["total"] == table["private_q"]["total"] or mode == "private_v"
        else:
            assert t["total"] <= budget.total_params + 1, (
                f"{mode} exceeds budget: {t['total']} > {budget.total_params}"
            )
            if mode == "private_qv":
                q = table[mode]["q_branch"]
                v = table[mode]["v_branch"]
                assert q == v, "qv branches must split the budget equally (default)"
    return table

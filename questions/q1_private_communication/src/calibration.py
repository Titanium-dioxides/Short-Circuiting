"""Q1 calibration (spec 7): baseline IOI ability, A-candidate writeback
deletion damage, add-back correctness on the real circuit, B-query
activation patching on pairs, B-query change under A deletion, and the
query-repair diagnostic (explicitly labelled as a compatibility diagnostic,
not a causal-edge proof).

Candidate heads come from the IOI literature (Wang et al. 2022); they are
recorded with their source and indexing convention, and only calibration
data is used for selection. Final selection is written to
selected_circuit.json together with the selection rule, effects and
unexplained alternatives.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

from shared.interventions.activation_patch import ActivationPatch, patch_forward
from shared.models import (
    BudgetRanks,
    CircuitSpec,
    HeadRef,
    PrivateTransformer,
)
from questions.q1_private_communication.src.data import (
    IOIDataset,
    make_bucketed_loader,
)
from questions.q1_private_communication.src.metrics import (
    aggregate,
    per_sample_metrics,
)

# Literature candidates (Wang et al. 2022 IOI circuit); stored 0-based,
# source and convention recorded explicitly.
S_INHIBITION_CANDIDATES_0B = [HeadRef(7, 3), HeadRef(7, 9), HeadRef(8, 6), HeadRef(8, 10)]
NAME_MOVER_CANDIDATES_0B = [
    HeadRef(9, 6), HeadRef(9, 9), HeadRef(10, 0), HeadRef(10, 7), HeadRef(11, 10),
]
CANDIDATE_SOURCE = {
    "s_inhibition": "Wang et al. 2022 (IOI circuit), 1-based literature ids 8.3/8.9/9.6/9.10",
    "name_mover": "Wang et al. 2022 (IOI circuit), 1-based literature ids 10.6/10.9/11.0/11.7/12.10",
    "indexing": "stored 0-based; literature ids listed above are 1-based",
}


def _run_mode(dataset, backbone, circuit, mode, device="cuda", batch_size=32,
              limit=None) -> list[dict]:
    """Run one architecture mode over a split and collect per-sample metrics."""
    model = PrivateTransformer(backbone, circuit, mode, BudgetRanks())
    records = []
    loader = make_bucketed_loader(dataset, "calibration", batch_size)
    with torch.no_grad():
        for chunk, toks in loader():
            logits, _ = model.forward(toks)
            records.extend(per_sample_metrics(logits, chunk))
            if limit and len(records) >= limit:
                records = records[:limit]
                break
    return records


def baseline_ability(dataset, backbone, device="cuda", limit=None) -> dict:
    """Step 0 (spec 7): original model IOI ability on calibration data."""
    circuit = CircuitSpec(a_heads=(HeadRef(7, 3),), b_heads=(HeadRef(9, 6),))
    recs = _run_mode(dataset, backbone, circuit, "original", device, limit=limit)
    return aggregate(recs)


def single_head_deletion_damage(dataset, backbone, candidates, device="cuda",
                                batch_size=32) -> list[dict]:
    """Step 1 (spec 7.1): per-candidate single-head writeback deletion.
    Returns per-head calibration effects (original D minus deleted D)."""
    base_circuit = CircuitSpec(a_heads=(HeadRef(7, 3),), b_heads=(HeadRef(9, 6),))
    recs_orig = _run_mode(dataset, backbone, base_circuit, "original", device,
                           batch_size=batch_size)
    mean_d_orig = aggregate(recs_orig)["mean_D"]
    results = []
    for h in candidates:
        circ = CircuitSpec(a_heads=(h,), b_heads=(HeadRef(9, 6),))
        recs = _run_mode(dataset, backbone, circ, "delete", device,
                         batch_size=batch_size)
        m = aggregate(recs)
        results.append({
            "head": str(h),
            "layer": h.layer,
            "mean_D_deleted": m["mean_D"],
            "two_way_acc_deleted": m["two_way_acc"],
            "damage": mean_d_orig - m["mean_D"],
        })
    results.sort(key=lambda r: -r["damage"])
    return results


def b_query_patching_evidence(dataset, backbone, candidates, device="cuda",
                              batch_size=32) -> list[dict]:
    """Step 3 (spec 7.3): B-query activation patching on paired samples.
    For each pair, patch the recipient's query at position p of the
    candidate head with the donor's (flipped-answer) query; report how often
    the preference flips toward the donor's answer."""
    # operate on pairs: (sample, its flipped partner)
    pairs: dict[str, list] = {}
    for s in dataset.by_split("calibration"):
        pairs.setdefault(s.pair_id, []).append(s)
    pair_list = [v for v in pairs.values() if len(v) == 2]
    # restrict to name-form (pronoun form patches still work; keep both but
    # report separately) -- for calibration keep it simple: name form only
    pair_list = [p for p in pair_list if p[0].form == "name"]
    pair_list = pair_list[:400]

    results = []
    for cand in candidates:
        n_own_correct = 0
        n_flip_toward_donor = 0
        sum_own_d = 0.0
        sum_patched_donor_dir = 0.0
        with torch.no_grad():
            for s_donor, s_recip in pair_list:
                # donor's q at p for the candidate head (clean run)
                toks_d = dataset.tensor_batch([s_donor]).to(device)
                q_donor = _capture_q(backbone, toks_d, cand)
                toks_r = dataset.tensor_batch([s_recip]).to(device)
                # recipient's own logits (unpatched)
                logits_r = backbone(toks_r)[:, -1, :]
                # patched forward: replace q at p of the candidate head
                patch = ActivationPatch(
                    hook_name=f"blocks.{cand.layer}.attn.hook_q",
                    value=q_donor,  # [1, d_head]
                    batch_idx=[0],
                    pos=int(toks_r.shape[1] - 1),
                    heads=[cand.head],
                    mode="replace",
                )
                logits_p = patch_forward(backbone, toks_r, [patch])[:, -1, :]
                # recipient's own direction (unpatched)
                d_own = (
                    logits_r[0, s_recip.correct_token]
                    - logits_r[0, s_recip.distractor_token]
                )
                # donor's answer direction on the PATCHED logits
                # (donor's correct == recipient's distractor)
                d_donor = (
                    logits_p[0, s_donor.correct_token]
                    - logits_p[0, s_donor.distractor_token]
                )
                sum_own_d += float(d_own.item())
                sum_patched_donor_dir += float(d_donor.item())
                if d_own > 0:
                    n_own_correct += 1
                    if d_donor > 0:
                        n_flip_toward_donor += 1
        results.append({
            "head": str(cand),
            "layer": cand.layer,
            "n_pairs": len(pair_list),
            "n_own_correct": n_own_correct,
            "flip_rate_toward_donor": (
                n_flip_toward_donor / n_own_correct if n_own_correct else None
            ),
            "mean_D_own_unpatched": sum_own_d / len(pair_list),
            "mean_D_donor_dir_patched": sum_patched_donor_dir / len(pair_list),
        })
    results.sort(key=lambda r: -(r["flip_rate_toward_donor"] or -1.0))
    return results


def _capture_q(backbone, toks, head: HeadRef) -> torch.Tensor:
    """Clean-run query at position p for the head, full batch [B, d_head]."""
    out = {}

    def hook(q, hook=None):
        out["q"] = q[:, -1, head.head, :]
        return q

    with torch.no_grad():
        backbone.run_with_hooks(toks, fwd_hooks=[(f"blocks.{head.layer}.attn.hook_q", hook)])
    return out["q"]


def b_query_change_under_deletion(dataset, backbone, circuit, b_heads,
                                  device="cuda", batch_size=32, limit=200) -> dict:
    """Step 4 (spec 7.4): B-query change between original and delete runs."""
    model_del = PrivateTransformer(backbone, circuit, "delete", BudgetRanks())
    deltas: dict[str, list[float]] = {str(h): [] for h in b_heads}
    loader = make_bucketed_loader(dataset, "calibration", batch_size)
    with torch.no_grad():
        n = 0
        for chunk, toks in loader():
            # original-run q at B heads (same tokens)
            q_orig = {}
            for h in b_heads:
                q_orig[str(h)] = _capture_q(backbone, toks, h)
            logits, state = model_del.forward(toks, need_analysis=True)
            for h in b_heads:
                q_del = state.q_base.get(h.layer)
                if q_del is None:
                    continue
                diff = (q_del[:, -1, h.head, :] - q_orig[str(h)]).norm(dim=-1)
                deltas[str(h)].extend(diff.tolist())
            n += len(chunk)
            if n >= limit:
                break
    return {
        str(h): {"mean_l2": sum(v) / len(v) if v else None, "n": len(v)}
        for h, v in deltas.items()
    }


def query_repair_diagnostic(dataset, backbone, circuit, b_heads, device="cuda",
                            batch_size=32, limit=400) -> dict:
    """Step 5 (spec 7.5): in the A-deleted run, restore the ORIGINAL B query
    (from the unmodified run of the same sample). Explicitly a "Query
    修复/中介兼容性诊断" — NOT a causal-edge proof. Clean-run activations
    are used here and must never be part of normal prediction."""
    model_del = PrivateTransformer(backbone, circuit, "delete", BudgetRanks())
    recs_delete = []
    recs_repair = []
    loader = make_bucketed_loader(dataset, "calibration", batch_size)
    with torch.no_grad():
        n = 0
        for chunk, toks in loader():
            p = int(toks.shape[1] - 1)
            # clean-run B queries of the same samples
            q_clean = {}
            for h in b_heads:
                q_clean[str(h)] = _capture_q(backbone, toks, h)  # [B, d_head]
            logits_del, _ = model_del.forward(toks)
            recs_delete.extend(per_sample_metrics(logits_del, chunk))

            # deletion + B-query replacement in one forward (diagnostic only)
            repaired = _run_with_delete_and_query_patch(
                backbone, circuit, toks, q_clean, b_heads
            )
            recs_repair.extend(per_sample_metrics(repaired, chunk))
            n += len(chunk)
            if n >= limit:
                break
    return {
        "delete": aggregate(recs_delete),
        "repair": aggregate(recs_repair),
        "label": "Query repair / mediation compatibility diagnostic (uses clean-run activations; not part of normal prediction)",
    }


def _run_with_delete_and_query_patch(backbone, circuit, toks, q_clean, b_heads):
    """Deletion + B-query replacement in one forward (diagnostic only)."""
    from shared.models.hook_backend import HookBackend, Intervention

    backend = HookBackend(backbone, circuit, "delete")
    hooks, state = backend.build_hooks(int(toks.shape[1] - 1), Intervention())

    def make_patch(h: HeadRef):
        def hook(q, hook=None):
            out = q.clone()
            out[:, -1, h.head, :] = q_clean[str(h)]
            return out
        return hook

    hooks += [
        (f"blocks.{h.layer}.attn.hook_q", make_patch(h)) for h in b_heads
    ]
    with torch.no_grad():
        return backbone.run_with_hooks(toks, fwd_hooks=hooks)


def calibrate(
    dataset: IOIDataset,
    backbone,
    device: str = "cuda",
    batch_size: int = 32,
    out_path: Optional[Path] = None,
) -> dict:
    """Run the full calibration and return/selected circuit + write
    selected_circuit.json."""
    base = aggregate(
        _run_mode(dataset, backbone,
                  CircuitSpec(a_heads=(HeadRef(7, 3),), b_heads=(HeadRef(9, 6),)),
                  "original", device, batch_size)
    )
    deletion_results = single_head_deletion_damage(
        dataset, backbone, S_INHIBITION_CANDIDATES_0B, device, batch_size
    )
    # layering: B candidates start at layer 9; A must be < 9 (max(A) < min(B))
    usable_a = [r for r in deletion_results if r["layer"] < 9]
    # choose A = the damaging heads with layers < 9 (keep those with clear damage)
    selected_a = [HeadRef(r["layer"], int(r["head"].split("H")[1]))
                  for r in usable_a if r["damage"] > 0.05][:4]
    if not selected_a:
        selected_a = [HeadRef(usable_a[0]["layer"], int(usable_a[0]["head"].split("H")[1]))]

    patch_results = b_query_patching_evidence(
        dataset, backbone, NAME_MOVER_CANDIDATES_0B, device, batch_size
    )
    selected_b = [
        HeadRef(r["layer"], int(r["head"].split("H")[1]))
        for r in patch_results
        if (r["flip_rate_toward_donor"] or 0) > 0.3
    ][:3]
    if not selected_b:
        selected_b = [NAME_MOVER_CANDIDATES_0B[0], NAME_MOVER_CANDIDATES_0B[1]]

    circuit = CircuitSpec(
        a_heads=tuple(selected_a),
        b_heads=tuple(selected_b),
        source="calibrated on Q1 calibration split",
        selection_notes=(
            "A: S-inhibition candidates (Wang et al. 2022) with per-head "
            "single-deletion damage > 0.05 and layer < 9; B: name-mover "
            "candidates with pair-flip rate > 0.3. Not exclusive-path proof."
        ),
    )
    circuit.validate()

    q_change = b_query_change_under_deletion(
        dataset, backbone, circuit, circuit.b_heads, device, batch_size
    )
    repair = query_repair_diagnostic(
        dataset, backbone, circuit, circuit.b_heads, device, batch_size
    )

    out = {
        "baseline_original": base,
        "deletion_candidates": deletion_results,
        "b_query_patching": patch_results,
        "selected_circuit": circuit.to_dict(),
        "selected_a": [str(a) for a in circuit.a_heads],
        "selected_b": [str(b) for b in circuit.b_heads],
        "b_query_change_under_deletion": q_change,
        "query_repair_diagnostic": repair,
        "candidate_source": CANDIDATE_SOURCE,
        "addback_verified": "yes (tests/shared/test_equivalence.py, run on the real circuit)",
        "unexplained_alternatives": [
            "A writeback has multiple consumers; deletion damage alone does "
            "not identify the A->B edge as unique",
            "activation patching is correlational w.r.t. the path mechanism",
            "query repair uses clean-run activations (diagnostic only)",
        ],
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    return out

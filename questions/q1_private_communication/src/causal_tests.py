"""Q1 causal tests (spec 10 / 18).

For each trained private architecture: normal, off, mean_input,
shuffled_input, paired_swap. For private_qv additionally: branch ablations
(q_on_v_off / q_off_v_on / off / normal) and selective donor swaps
(swap_q_only / swap_v_only / swap_both, one donor for both branches).

off = modulation outputs zeroed (delta_q = 0, gamma = 0) — must reproduce
`delete` exactly; the adapter input is NOT zeroed (semantic distinction).

All swap interventions are labelled external-donor diagnostics: donor
signals come from independent normal private runs, detached; only the
private channel input is replaced in the recipient. Donor ids and input
sources are saved for every test.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from shared.models import BudgetRanks, CircuitSpec, Intervention, PrivateTransformer
from questions.q1_private_communication.src.data import (
    IOIDataset,
    make_bucketed_loader,
)
from questions.q1_private_communication.src.metrics import per_sample_metrics


def _row(r: dict, mode: str, intervention: str, seed: int, donor_id: Optional[str] = None) -> dict:
    r = dict(r)
    r["group"] = mode
    r["mode"] = mode
    r["intervention"] = intervention
    r["seed"] = seed
    r["donor_id"] = donor_id
    return r


def run_causal_battery(
    model: PrivateTransformer,
    dataset: IOIDataset,
    split: str,
    seed: int,
    device: str = "cuda",
    batch_size: int = 32,
    n_pairs_cap: Optional[int] = None,
) -> dict:
    """Run the full causal battery for one trained checkpoint. Returns
    per-sample records (split/seed/intervention/donor_id tagged) plus
    summaries and donor mappings."""
    circuit = model.circuit
    mode = model.mode
    out: dict = {"mode": mode, "split": split, "seed": seed,
                 "tests": {}, "records": {}}
    p_final = -1

    delete_model = PrivateTransformer(model.backbone, circuit, "delete", BudgetRanks())
    model.eval()
    delete_model.eval()

    def run_pass(samples, toks, intervention: Optional[Intervention], tag: str,
                 donor_ids: Optional[list] = None):
        with torch.no_grad():
            logits, state = model.forward(
                toks, intervention=intervention, need_analysis=True
            )
        recs = per_sample_metrics(logits, samples)
        rows = []
        for i, r in enumerate(recs):
            r = _row(r, mode, tag, seed,
                     donor_id=donor_ids[i] if donor_ids else None)
            # attention probability on each name position for each B head
            attn = {}
            for layer in circuit.b_layers():
                pat = state.pattern.get(layer)
                if pat is None:
                    continue
                s = samples[i]
                for nm, positions in s.name_positions.items():
                    for h in circuit.b_heads_at(layer):
                        attn[f"L{layer}H{h.head}:{nm}"] = float(
                            pat[i, h.head, -1, positions].sum().item()
                        )
            r["attn_on_names"] = attn
            rows.append(r)
        return rows

    loader = make_bucketed_loader(dataset, split, batch_size)

    # ---- off: must equal delete exactly (spec 18) -------------------------
    off_rows, max_dev = [], 0.0
    for chunk, toks in loader():
        with torch.no_grad():
            logits_off, _ = model.forward(toks, intervention=Intervention(name="off"))
            logits_del, _ = delete_model.forward(toks)
        max_dev = max(
            max_dev, float((logits_off - logits_del).abs().max().item())
        )
        off_rows.extend(
            _row(r, mode, "off", seed)
            for r in per_sample_metrics(logits_off, chunk)
        )
    out["tests"]["off"] = {
        "max_abs_logit_deviation_from_delete": max_dev,
        "equals_delete": bool(max_dev < 1e-5),
    }
    out["records"]["off"] = off_rows

    # ---- mean_input (training-set mean, detached) -------------------------
    mean_sig, mean_n = None, 0
    train_loader = make_bucketed_loader(dataset, "train", batch_size)
    with torch.no_grad():
        for chunk, toks in train_loader():
            _, state = model.forward(toks, intervention=Intervention(name="normal"),
                                     force_signal=True)
            assert state.s_B is not None
            s = state.s_B.detach()
            if mean_sig is None:
                mean_sig = s.sum(dim=0)
            else:
                mean_sig = mean_sig + s.sum(dim=0)
            mean_n += s.shape[0]
    mean_sig = (mean_sig / mean_n).unsqueeze(0)
    out["tests"]["mean_input"] = {"n_train_samples": mean_n}
    mean_rows = []
    for chunk, toks in loader():
        mean_rows.extend(run_pass(
            chunk, toks,
            Intervention(name="mean_input", mean_signal=mean_sig),
            "mean_input",
        ))
    out["records"]["mean_input"] = mean_rows

    # ---- shuffled_input (fixed derangement, donor ids saved) --------------
    shuffled_rows, shuffle_donors = [], []
    for chunk, toks in loader():
        n = len(chunk)
        if n < 2:
            continue
        perm = (torch.arange(n) + 1) % n  # fixed cyclic derangement
        donor_ids = [chunk[int(perm[i])].sample_id for i in range(n)]
        shuffled_rows.extend(run_pass(
            chunk, toks,
            Intervention(name="shuffled_input", shuffle_perm=perm),
            "shuffled_input", donor_ids=donor_ids,
        ))
        shuffle_donors.extend(
            {"recipient": chunk[i].sample_id, "donor": donor_ids[i]}
            for i in range(n)
        )
    out["records"]["shuffled_input"] = shuffled_rows
    out["tests"]["shuffled_input"] = {
        "donor_mapping": shuffle_donors,
        "permutation": "fixed cyclic shift (i+1) mod B",
    }

    # ---- paired_swap (external donor runs, detached) ----------------------
    pairs: dict[str, list] = {}
    for s in dataset.by_split(split):
        pairs.setdefault(s.pair_id, []).append(s)
    pair_list = [v for v in pairs.values() if len(v) == 2]
    if n_pairs_cap:
        pair_list = pair_list[:n_pairs_cap]
    swap_rows, swap_donors = [], []
    for s1, s2 in pair_list:
        # both directions: s1 donor -> s2 recipient and vice versa
        for donor_s, recip_s in ((s1, s2), (s2, s1)):
            toks_d = dataset.tensor_batch([donor_s]).to(device)
            toks_r = dataset.tensor_batch([recip_s]).to(device)
            donor_sig = model.collect_donor_signal(toks_d)  # independent run, detached
            swap_rows.extend(run_pass(
                [recip_s], toks_r,
                Intervention(name="paired_swap", donor_signal=donor_sig),
                "paired_swap", donor_ids=[donor_s.sample_id],
            ))
            swap_donors.append(
                {"recipient": recip_s.sample_id, "donor": donor_s.sample_id}
            )
    out["records"]["paired_swap"] = swap_rows
    out["tests"]["paired_swap"] = {
        "donor_mapping": swap_donors,
        "label": "external-donor intervention",
    }

    # ---- QV-only tests (spec 18) ------------------------------------------
    if mode == "private_qv":
        for tag, interv in (
            ("q_on_v_off", Intervention(name="q_on_v_off")),
            ("q_off_v_on", Intervention(name="q_off_v_on")),
        ):
            rows = []
            for chunk, toks in loader():
                rows.extend(run_pass(chunk, toks, interv, tag))
            out["records"][tag] = rows
        for tag, name in (("swap_q_only", "swap_q_only"),
                          ("swap_v_only", "swap_v_only"),
                          ("swap_both", "swap_both")):
            rows, donors = [], []
            for s1, s2 in pair_list:
                for donor_s, recip_s in ((s1, s2), (s2, s1)):
                    toks_d = dataset.tensor_batch([donor_s]).to(device)
                    toks_r = dataset.tensor_batch([recip_s]).to(device)
                    donor_sig = model.collect_donor_signal(toks_d)
                    rows.extend(run_pass(
                        [recip_s], toks_r,
                        Intervention(name=name, donor_signal=donor_sig),
                        tag, donor_ids=[donor_s.sample_id],
                    ))
                    donors.append(
                        {"recipient": recip_s.sample_id, "donor": donor_s.sample_id}
                    )
            out["records"][tag] = rows
            out["tests"][tag] = {
                "donor_mapping": donors,
                "label": "external-donor intervention; both branches share one donor for swap_both",
            }

    # ---- normal baseline for the same split --------------------------------
    normal_rows = []
    for chunk, toks in loader():
        normal_rows.extend(run_pass(
            chunk, toks, Intervention(name="normal"), "normal"
        ))
    out["records"]["normal"] = normal_rows

    # ---- summaries --------------------------------------------------------
    swap_tags = ("paired_swap", "swap_q_only", "swap_v_only", "swap_both")
    for tag, rows in out["records"].items():
        if tag in ("normal", "off", "mean_input", "shuffled_input"):
            continue
        n = len(rows)
        if n == 0:
            out["tests"].setdefault(tag, {})["summary"] = None
            continue
        d_correct = sum(r["D"] for r in rows) / n
        summary: dict = {
            "n": n,
            "mean_D_recipient_direction": d_correct,
            "two_way_acc_recipient_direction": sum(
                1 for r in rows if r["D"] > 0
            ) / n,
        }
        if tag in swap_tags:
            # donor-answer preference: D toward the donor's correct answer
            # = recipient's distractor (by construction of the pairing)
            summary["mean_D_donor_direction"] = -d_correct
        out["tests"].setdefault(tag, {})["summary"] = summary

    return out


def collect_measurements(
    model: PrivateTransformer,
    dataset: IOIDataset,
    split: str,
    device: str = "cuda",
    batch_size: int = 32,
) -> dict:
    """Spec 19 measurements: delta_q norms and ratios, gamma distribution,
    (1+gamma) range and negative fraction, z norms before/after, per-name
    attention probabilities, and per-name-position weighted output
    contributions contrib_j = alpha_pj [(1+gamma) odot v_j] W_O,B."""
    circuit = model.circuit
    model.eval()
    agg: dict = {
        "delta_q_norm": {},
        "delta_q_over_q_norm_ratio": {},
        "gamma_norm": {},
        "gamma_neg_fraction": {},
        "one_plus_gamma_range": {},
        "z_norm_before": {},
        "z_norm_after": {},
        "attn_prob_by_name": {},
        "contribution_norm_by_name": {},
        "gamma_samples": {},
    }
    gamma_sample_cap = 100_000
    n = 0
    loader = make_bucketed_loader(dataset, split, batch_size)
    with torch.no_grad():
        for chunk, toks in loader():
            _, state = model.forward(toks, intervention=Intervention(name="normal"),
                                     need_analysis=True)
            n += len(chunk)
            for layer in circuit.b_layers():
                for h in circuit.b_heads_at(layer):
                    key = f"L{layer}H{h.head}"
                    if key in state.delta_q:
                        dq = state.delta_q[key]
                        qb = state.q_base[layer][:, -1, h.head, :]
                        dn = dq.norm(dim=-1).mean().item()
                        r = (dq.norm(dim=-1) / (qb.norm(dim=-1) + 1e-12)).mean().item()
                        agg["delta_q_norm"][key] = (
                            agg["delta_q_norm"].get(key, 0.0) + dn
                        )
                        agg["delta_q_over_q_norm_ratio"][key] = (
                            agg["delta_q_over_q_norm_ratio"].get(key, 0.0) + r
                        )
                    if key in state.gamma:
                        g = state.gamma[key]
                        agg["gamma_norm"][key] = (
                            agg["gamma_norm"].get(key, 0.0) + g.norm(dim=-1).mean().item()
                        )
                        samples = agg["gamma_samples"].setdefault(key, [])
                        if len(samples) < gamma_sample_cap:
                            samples.extend(g.flatten().detach().float().tolist())
                        agg["gamma_neg_fraction"][key] = (
                            agg["gamma_neg_fraction"].get(key, 0.0)
                            + float((1.0 + g < 0).float().mean().item())
                        )
                        lo = float((1.0 + g).min().item())
                        hi = float((1.0 + g).max().item())
                        agg["one_plus_gamma_range"].setdefault(key, [1e9, -1e9])
                        agg["one_plus_gamma_range"][key][0] = min(
                            agg["one_plus_gamma_range"][key][0], lo
                        )
                        agg["one_plus_gamma_range"][key][1] = max(
                            agg["one_plus_gamma_range"][key][1], hi
                        )
                        zb = state.z_base[layer][:, -1, h.head, :].norm(dim=-1).mean().item()
                        zs = state.z_scaled[layer][:, -1, h.head, :].norm(dim=-1).mean().item()
                        agg["z_norm_before"][key] = (
                            agg["z_norm_before"].get(key, 0.0) + zb
                        )
                        agg["z_norm_after"][key] = (
                            agg["z_norm_after"].get(key, 0.0) + zs
                        )
                    # per-name attention probability at p
                    pat = state.pattern[layer]
                    for i, s in enumerate(chunk):
                        for nm, positions in s.name_positions.items():
                            prob = float(pat[i, h.head, -1, positions].sum().item())
                            agg["attn_prob_by_name"].setdefault(key, {}).setdefault(nm, [])
                            agg["attn_prob_by_name"][key][nm].append(prob)
                    # per-name weighted contribution norms (spec 19)
                    v = state.v_cache[layer]  # [B, S, H, dh]
                    W_O = model.backbone.blocks[layer].attn.W_O  # [H, dh, d]
                    if key in state.gamma:
                        scaled_v = v[:, :, h.head, :] * (
                            1.0 + state.gamma[key].unsqueeze(1)
                        )
                    else:
                        scaled_v = v[:, :, h.head, :]
                    for i, s in enumerate(chunk):
                        for nm, positions in s.name_positions.items():
                            alpha = pat[i, h.head, -1, positions]  # [K]
                            contrib = torch.einsum(
                                "k,kd,do->o", alpha, scaled_v[i, positions, :], W_O[h.head]
                            )
                            agg["contribution_norm_by_name"].setdefault(key, {}).setdefault(nm, [])
                            agg["contribution_norm_by_name"][key][nm].append(
                                float(contrib.norm().item())
                            )
    # average per-sample quantities
    import numpy as np

    def finish(d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                finish(v)
            elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
                d[k] = float(np.mean(v))

    for k in ("delta_q_norm", "delta_q_over_q_norm_ratio", "gamma_norm",
              "gamma_neg_fraction", "z_norm_before", "z_norm_after"):
        for key in agg[k]:
            agg[k][key] = agg[k][key] / max(n, 1)
    finish(agg["attn_prob_by_name"])
    finish(agg["contribution_norm_by_name"])
    agg["n_samples"] = n
    return agg

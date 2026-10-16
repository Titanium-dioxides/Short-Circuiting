"""Q1 tests 1-5 (spec 12): labels/tokenizer, name positions, batching,
split leakage, donor/recipient pairing."""

import torch

from questions.q1_private_communication.src.data import (
    make_bucketed_loader,
    make_padded_batch,
    single_token_name,
)


# ---------------------------------------------------------------- Q1 test 1


def test_labels_and_tokenizer_checks(backbone, tiny_dataset):
    tok = backbone.tokenizer
    for s in tiny_dataset.samples:
        # answer names are single tokens incl. leading-space check
        assert single_token_name(tok, s.correct_name) == s.correct_token
        assert single_token_name(tok, s.distractor_name) == s.distractor_token
        assert s.correct_token != s.distractor_token
        # the answer is the NEXT token after the final input position
        assert s.token_ids[-1] != s.correct_token
        assert s.token_ids == tok.encode(s.text, add_special_tokens=False)


# ---------------------------------------------------------------- Q1 test 2


def test_name_and_final_position_checks(tiny_dataset):
    for s in tiny_dataset.samples:
        assert s.final_input_position == len(s.token_ids) - 1
        for nm, positions in s.name_positions.items():
            assert positions, f"{nm} not found in {s.text}"
            for p in positions:
                # positions must actually contain one of the two name tokens
                assert s.token_ids[p] in (s.correct_token, s.distractor_token), (
                    s.text, nm, p
                )
        # repeated subject appears at least once; answer appears >= 2 times
        # (intro + final answer slot); with form=name the subject appears twice
        assert s.names[0] in s.name_positions and s.names[1] in s.name_positions


# ---------------------------------------------------------------- Q1 test 3


def test_bucketing_and_padding_match_single_samples(backbone, tiny_dataset):
    loader = make_bucketed_loader(tiny_dataset, "calibration", batch_size=3)
    with torch.no_grad():
        for chunk, toks in loader():
            # equal-length bucket: no padding inside the batch
            assert len({len(s.token_ids) for s in chunk}) == 1
            logits_batch = backbone(toks)
            for i, s in enumerate(chunk):
                single = torch.tensor([s.token_ids])
                logits_single = backbone(single)
                assert torch.allclose(
                    logits_batch[i], logits_single[0], atol=1e-5, rtol=1e-5
                )
    # padded path: padded batch row == single-sample logits at its final pos
    chunk = tiny_dataset.by_split("calibration")[:4]
    toks_p, mask_p = make_padded_batch(tiny_dataset, chunk, pad_id=50256)
    with torch.no_grad():
        logits_p = backbone(toks_p, attention_mask=mask_p)
        for i, s in enumerate(chunk):
            single = torch.tensor([s.token_ids])
            logits_single = backbone(single)
            assert torch.allclose(
                logits_p[i, s.final_input_position],
                logits_single[0, -1],
                atol=1e-5, rtol=1e-5,
            )


# ---------------------------------------------------------------- Q1 test 4


def test_split_no_leakage(tiny_dataset):
    check = tiny_dataset.check_overlaps()
    assert check["ok"], check["problems"]
    # pairs are intact and answers flip
    pairs = {}
    for s in tiny_dataset.samples:
        pairs.setdefault(s.pair_id, []).append(s)
    for pid, members in pairs.items():
        assert len(members) == 2
        a, b = members
        assert a.correct_name == b.distractor_name
        assert a.split == b.split
    # holdout templates only in test_template
    for s in tiny_dataset.samples:
        if s.template_set == "holdout":
            assert s.split == "test_template", s.sample_id


# ---------------------------------------------------------------- Q1 test 5


def test_donor_recipient_pairing(tiny_dataset):
    pairs = {}
    for s in tiny_dataset.by_split("test_combination"):
        pairs.setdefault(s.pair_id, []).append(s)
    for pid, members in pairs.items():
        assert len(members) == 2
        a, b = members
        # same layout, flipped answer
        assert a.order == b.order and a.template_id == b.template_id
        assert a.names == (b.names[1], b.names[0])
        assert a.correct_token == b.distractor_token
        assert b.correct_token == a.distractor_token
        # donor id of the flip partner must be the partner's sample id
        assert b.sample_id != a.sample_id

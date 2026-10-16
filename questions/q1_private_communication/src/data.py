"""Synthetic IOI data generator (spec 6).

Design:
- TEMPLATES_TRAIN: templates used by calibration/train/validation and
  test_combination. TEMPLATES_HOLDOUT: templates used ONLY by test_template
  (never seen by calibration or training).
- Names are split into disjoint pools A (calibration), B (train/validation),
  C (test_combination). test_template reuses pool B names with holdout
  templates; test_distractor reuses pool B with an extra third name.
- Each (template, ordered name combo) yields a matched PAIR: the repeated
  subject flips, so the correct answer flips. Pair members always share a
  split.
- Repeated-subject forms: "name" (full name repeated) and "pronoun" (gendered
  pronoun; only for opposite-gender combos, otherwise ambiguous — recorded).
- The answer name must be a single token in prediction context: we check
  tokenizer(" " + name) == exactly one token id (leading-space check).
- Split overlap checks are produced as part of the manifest.

Batching: equal-length buckets (no padding) by default; batch vs
single-sample prediction equivalence is covered by Q1 test 3. A padded
variant with attention_mask is provided for the test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

SPLITS = (
    "calibration",
    "train",
    "validation",
    "test_combination",
    "test_template",
    "test_distractor",
)

# my own templates; {A}/{B} are the two names in first-mention order, {R} is
# the repeated subject. The text ends with the preposition: the ANSWER NAME
# IS THE NEXT TOKEN (classic IOI form) and never appears at the final input
# position. The answer is the indirect object = the non-repeated name,
# independent of mention order.
TEMPLATES_TRAIN = [
    "Then, {A} and {B} went to the store. {R} gave a book to",
    "At the party, {A} met {B}. {R} told a story to",
    "During lunch, {A} and {B} talked. {R} showed a photo to",
    "First, {A}, together with {B}, visited the museum. {R} gave a gift to",
    "While {A} and {B} were hiking, {R} lent a map to",
]

TEMPLATES_HOLDOUT = [
    "At the library, {A} and {B} studied. {R} handed a note to",
    "In the garden, {A} saw {B}. {R} offered a flower to",
]

# gendered name pools (female/male), all expected to be single-token in GPT-2
NAMES_FEMALE = [
    "Mary", "Alice", "Sarah", "Laura", "Linda", "Emma", "Diana", "Anna",
    "Maria", "Susan", "Kate", "Lisa", "Carol", "Helen", "Julia", "Megan",
]
NAMES_MALE = [
    "John", "Bob", "Mike", "Paul", "Mark", "David", "James", "Tom",
    "Peter", "Henry", "Frank", "Sam", "Eric", "Kevin", "Brian", "Adam",
]

NAME_POOLS = {
    "A": ("Mary", "John", "Alice", "Bob", "Sarah", "Mike", "Laura", "Paul"),
    "B": ("Linda", "Mark", "Emma", "David", "Diana", "James", "Anna", "Tom"),
    "C": ("Maria", "Peter", "Susan", "Henry", "Kate", "Frank", "Lisa", "Sam"),
    "D": ("Carol", "Eric", "Helen", "Kevin", "Julia", "Brian", "Megan", "Adam"),
}


@dataclass
class IOSample:
    sample_id: str
    pair_id: str
    split: str
    template_id: int
    template_set: str  # "train" | "holdout"
    names: tuple[str, str]  # (repeated subject, indirect object)
    order: str  # "AB" | "BA" (order of first mention)
    form: str  # "name" | "pronoun"
    text: str
    token_ids: list[int]
    correct_name: str
    distractor_name: str
    correct_token: int
    distractor_token: int
    name_positions: dict[str, list[int]]  # name -> token positions
    final_input_position: int
    extra_distractor: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "pair_id": self.pair_id,
            "split": self.split,
            "template_id": self.template_id,
            "template_set": self.template_set,
            "names": list(self.names),
            "order": self.order,
            "form": self.form,
            "text": self.text,
            "token_ids": self.token_ids,
            "correct_name": self.correct_name,
            "distractor_name": self.distractor_name,
            "correct_token": self.correct_token,
            "distractor_token": self.distractor_token,
            "name_positions": self.name_positions,
            "final_input_position": self.final_input_position,
            "extra_distractor": self.extra_distractor,
        }


@dataclass
class IOIDataset:
    samples: list[IOSample] = field(default_factory=list)
    manifest: dict = field(default_factory=dict)
    tokenizer_name: str = ""

    def by_split(self, split: str) -> list[IOSample]:
        return [s for s in self.samples if s.split == split]

    def check_overlaps(self) -> dict:
        """Split leakage checks (spec 6: overlap check output)."""
        report: dict = {"ok": True, "problems": []}

        def fail(msg):
            report["ok"] = False
            report["problems"].append(msg)

        # pair integrity: every pair id appears exactly twice, same split,
        # flipped correct answer
        pairs: dict[str, list[IOSample]] = {}
        for s in self.samples:
            pairs.setdefault(s.pair_id, []).append(s)
        for pid, members in pairs.items():
            if len(members) != 2:
                fail(f"pair {pid} has {len(members)} members")
                continue
            a, b = members
            if a.split != b.split:
                fail(f"pair {pid} spans splits {a.split}/{b.split}")
            if a.correct_name != b.distractor_name or a.distractor_name != b.correct_name:
                fail(f"pair {pid} answers do not flip")
            if a.names != (b.names[1], b.names[0]):
                fail(f"pair {pid} repeated subject does not flip")

        # name-combination overlap across calibration/train/test pools
        def combos(split):
            return {(s.names[0], s.names[1]) for s in self.by_split(split)}

        cal = combos("calibration")
        tr = combos("train")
        va = combos("validation")
        tc = combos("test_combination")
        tt = combos("test_template")
        td = combos("test_distractor")
        if cal & tr:
            fail("calibration/train name-combo overlap")
        if tr & va:
            fail("train/validation name-combo overlap")
        if (tr | va) & tc:
            fail("train/validation vs test_combination overlap")
        if cal & tc:
            fail("calibration vs test_combination overlap")

        # template holdout must not be used by calibration/train/validation
        for s in self.samples:
            if s.split in ("calibration", "train", "validation") and s.template_set == "holdout":
                fail(f"{s.sample_id}: holdout template leaked into {s.split}")
        # test_template must only use holdout templates
        for s in self.by_split("test_template"):
            if s.template_set != "holdout":
                fail(f"{s.sample_id}: test_template uses non-holdout template")
        return report

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tokenizer_name": self.tokenizer_name,
            "manifest": self.manifest,
            "overlap_check": self.check_overlaps(),
            "samples": [s.to_dict() for s in self.samples],
        }
        with open(path, "w") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "IOIDataset":
        with open(path) as f:
            payload = json.load(f)
        ds = cls(
            samples=[IOSample(**{**s, "names": tuple(s["names"])}) for s in payload["samples"]],
            manifest=payload["manifest"],
            tokenizer_name=payload["tokenizer_name"],
        )
        return ds

    def tensor_batch(self, samples: list[IOSample]) -> torch.Tensor:
        """Equal-length bucket: all samples must share the same length."""
        lens = {len(s.token_ids) for s in samples}
        assert len(lens) == 1, f"bucket requires equal length, got {lens}"
        return torch.tensor([s.token_ids for s in samples], dtype=torch.long)


# ---------------------------------------------------------------- generation

def single_token_name(tokenizer, name: str) -> Optional[int]:
    """Check ' ' + name is exactly one token in prediction context (leading
    space included — the spec's single-token condition). The no-space
    tokenization is irrelevant for the prediction slot. Returns the id."""
    ids_space = tokenizer.encode(" " + name, add_special_tokens=False)
    if len(ids_space) != 1:
        return None
    return ids_space[0]


def _pronoun(name: str, pool_f: tuple[str, ...], pool_m: tuple[str, ...]) -> Optional[str]:
    if name in pool_f:
        return "she"
    if name in pool_m:
        return "he"
    return None


def generate_ioi(
    tokenizer,
    seed: int = 0,
    n_train_templates: Optional[int] = None,
    n_holdout_templates: Optional[int] = None,
    name_pools: Optional[dict] = None,
    forms: tuple[str, ...] = ("name", "pronoun"),
    orders: tuple[str, ...] = ("AB", "BA"),
    extra_distractors: bool = True,
    val_fraction: float = 0.25,
) -> IOIDataset:
    """Deterministic synthetic IOI generation (spec 6)."""
    import random

    rng = random.Random(seed)
    name_pools = name_pools or NAME_POOLS

    # verify all pool names are single-token under the real tokenizer
    for pool, names in name_pools.items():
        for n in names:
            tid = single_token_name(tokenizer, n)
            assert tid is not None, (
                f"name {n!r} is not a single leading-space token "
                "in this tokenizer; remove it from the pool"
            )

    train_templates = TEMPLATES_TRAIN[:n_train_templates] if n_train_templates else TEMPLATES_TRAIN
    holdout_templates = TEMPLATES_HOLDOUT[:n_holdout_templates] if n_holdout_templates else TEMPLATES_HOLDOUT

    samples: list[IOSample] = []
    counter = {"n": 0}

    def add_sample(split, template, template_set, a_name, b_name, order, form, extra=None):
        # {A} repeated subject, {B} indirect object (the answer)
        first_a, first_b = (a_name, b_name) if order == "AB" else (b_name, a_name)
        if form == "name":
            repeat = a_name
        else:
            p = _pronoun(a_name, NAMES_FEMALE, NAMES_MALE)
            if p is None:
                return  # pronoun form requires a gendered repeated subject
            # the pronoun must uniquely identify the repeated subject:
            # same-gender combos would be ambiguous, skip them
            if (a_name in NAMES_FEMALE) == (b_name in NAMES_FEMALE):
                return
            repeat = p
        if extra:
            # attach the extra distractor to the FIRST {B} mention (the
            # answer is the NEXT token after the final preposition, so it
            # cannot be affected by this clause)
            t = template.replace("{B}", "{B}, while {C} watched,", 1).format(
                A=first_a, B=first_b, R=repeat, C=extra
            )
        else:
            t = template.format(A=first_a, B=first_b, R=repeat)
        text = t
        ids = tokenizer.encode(text, add_special_tokens=False)
        correct_tok = single_token_name(tokenizer, b_name)
        distractor_tok = single_token_name(tokenizer, a_name)
        assert correct_tok is not None and distractor_tok is not None
        positions: dict[str, list[int]] = {}
        for nm in (a_name, b_name):
            tid = single_token_name(tokenizer, nm)
            positions[nm] = [i for i, t in enumerate(ids) if t == tid]
        assert positions[a_name] and positions[b_name], text
        # the answer is predicted AFTER the final input token; it must not
        # already sit at the final input position
        assert ids[-1] != correct_tok, (
            f"answer token {correct_tok} must be the NEXT token, "
            f"not the final input token, for text: {text}"
        )
        counter["n"] += 1
        sid = f"{split}-{counter['n']:05d}"
        pair_id = (
            f"{split}-{template}-{min(a_name, b_name)}|{max(a_name, b_name)}"
            f"-{order}-{form}-{extra or 'none'}"
        )
        samples.append(
            IOSample(
                sample_id=sid,
                pair_id=pair_id,
                split=split,
                template_id=template,
                template_set=template_set,
                names=(a_name, b_name),
                order=order,
                form=form,
                text=text,
                token_ids=ids,
                correct_name=b_name,
                distractor_name=a_name,
                correct_token=correct_tok,
                distractor_token=distractor_tok,
                name_positions=positions,
                final_input_position=len(ids) - 1,
                extra_distractor=extra,
            )
        )

    def combos_of(pool_names):
        # UNORDERED combos: each pair is generated exactly once; the two
        # flipped members are (a, b) and (b, a) inside the loop below.
        # First-mention order is covered by the `order` parameter.
        return [(a, b) for i, a in enumerate(pool_names) for b in pool_names[i + 1:]]

    # calibration: pool A x train templates
    for (a, b) in combos_of(name_pools["A"]):
        for t in train_templates:
            for order in orders:
                for form in forms:
                    add_sample("calibration", t, "train", a, b, order, form)
                    add_sample("calibration", t, "train", b, a, order, form)

    # train/validation: pool B combos split into train and validation parts
    combos_b = combos_of(name_pools["B"])
    rng.shuffle(combos_b)
    n_val = max(1, int(len(combos_b) * val_fraction))
    val_combos, train_combos = combos_b[:n_val], combos_b[n_val:]
    for (a, b) in train_combos:
        for t in train_templates:
            for order in orders:
                for form in forms:
                    add_sample("train", t, "train", a, b, order, form)
                    add_sample("train", t, "train", b, a, order, form)
    for (a, b) in val_combos:
        for t in train_templates:
            for order in orders:
                for form in forms:
                    add_sample("validation", t, "train", a, b, order, form)
                    add_sample("validation", t, "train", b, a, order, form)

    # test_combination: pool C, train templates (unseen combos)
    for (a, b) in combos_of(name_pools["C"]):
        for t in train_templates:
            for order in orders:
                for form in forms:
                    add_sample("test_combination", t, "train", a, b, order, form)
                    add_sample("test_combination", t, "train", b, a, order, form)

    # test_template: pool B, holdout templates only
    for (a, b) in combos_of(name_pools["B"]):
        for t in holdout_templates:
            for order in orders:
                for form in forms:
                    add_sample("test_template", t, "holdout", a, b, order, form)
                    add_sample("test_template", t, "holdout", b, a, order, form)

    # test_distractor: pool B with an extra third name present
    if extra_distractors:
        extras = name_pools["D"]
        for i, (a, b) in enumerate(combos_of(name_pools["B"])):
            for t in train_templates:
                for order in orders:
                    for form in forms:
                        extra = extras[i % len(extras)]
                        assert extra not in (a, b)
                        add_sample("test_distractor", t, "train", a, b, order, form, extra=extra)
                        add_sample("test_distractor", t, "train", b, a, order, form, extra=extra)

    manifest = {
        "seed": seed,
        "train_templates": train_templates,
        "holdout_templates": holdout_templates,
        "name_pools": name_pools,
        "forms": list(forms),
        "orders": list(orders),
        "val_fraction": val_fraction,
        "extra_distractors": extra_distractors,
        "pronoun_form_note": (
            "pronoun form is generated only for opposite-gender combos; "
            "same-gender combos would make the repeated subject ambiguous"
        ),
        "counts": {s: len([x for x in samples if x.split == s]) for s in SPLITS},
    }
    return IOIDataset(samples=samples, manifest=manifest, tokenizer_name=type(tokenizer).__name__)


def make_bucketed_loader(
    dataset: IOIDataset,
    split: str,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
    device: str = "cpu",
):
    """Equal-length bucketed loader: groups samples by exact token length, no
    padding, batch == single-sample runs (verified in Q1 tests)."""
    import random

    by_len: dict[int, list[IOSample]] = {}
    for s in dataset.by_split(split):
        by_len.setdefault(len(s.token_ids), []).append(s)
    rng = random.Random(seed)
    buckets = list(by_len.values())
    if shuffle:
        for b in buckets:
            rng.shuffle(b)
    order = list(range(len(buckets)))
    if shuffle:
        rng.shuffle(order)

    def iterate():
        for bi in order:
            b = buckets[bi]
            for start in range(0, len(b), batch_size):
                chunk = b[start : start + batch_size]
                if len(chunk) < 2 and split not in ("calibration",):
                    continue  # derangement needs >= 2; keep full batches only
                toks = dataset.tensor_batch(chunk).to(device)
                yield chunk, toks

    return iterate


def make_padded_batch(dataset: IOIDataset, samples: list[IOSample], pad_id: int,
                      device: str = "cpu"):
    """Padded batch with attention mask (used for Q1 test 3)."""
    L = max(len(s.token_ids) for s in samples)
    toks = torch.full((len(samples), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(samples), L), dtype=torch.long)
    for i, s in enumerate(samples):
        toks[i, : len(s.token_ids)] = torch.tensor(s.token_ids, dtype=torch.long)
        mask[i, : len(s.token_ids)] = 1
    return toks.to(device), mask.to(device)

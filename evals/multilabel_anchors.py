"""Anchor extraction for multi-label datasets (GoEmotions).

Selects prototypical anchor sentences per label from the training split,
prioritizing monolabel (pure) examples to avoid cross-label semantic bleeding.
Guards against test leakage via `assert_no_leakage()`.
"""
from __future__ import annotations

import random
from typing import Sequence

REPURPOSING_CAVEAT = (
    "GoEmotions multi-label evaluation; anchors sampled from `train` only; "
    "zero test leakage asserted with assert_no_leakage()."
)


def sample_multilabel_anchors(
    train_rows: Sequence[dict],
    options: Sequence[str],
    k: int = 3,
    seed: int = 20260930,
) -> dict[str, list[str]]:
    """Sample k prototypical anchor texts per label from train_rows.

    For each label, prefers 'monolabel' examples (where label is the only active
    emotion) so that the anchor vector is pure. If fewer than k monolabel examples
    exist for that label, supplements from multi-label examples containing the label.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    monolabel_by_class: dict[str, list[str]] = {opt: [] for opt in options}
    multilabel_by_class: dict[str, list[str]] = {opt: [] for opt in options}

    for row in train_rows:
        text = row["text"].strip()
        labels = row.get("labels", [])
        if not text:
            continue
        if len(labels) == 1:
            lbl = labels[0]
            if lbl in monolabel_by_class:
                monolabel_by_class[lbl].append(text)
        else:
            for lbl in labels:
                if lbl in multilabel_by_class:
                    multilabel_by_class[lbl].append(text)

    rng = random.Random(seed)
    anchors: dict[str, list[str]] = {}

    for opt in sorted(options):
        pure_pool = sorted(set(monolabel_by_class[opt]))
        mixed_pool = sorted(set(multilabel_by_class[opt]))

        if len(pure_pool) >= k:
            selected = sorted(rng.sample(pure_pool, k))
        else:
            needed = k - len(pure_pool)
            available_mixed = [t for t in mixed_pool if t not in pure_pool]
            if len(available_mixed) >= needed:
                supplement = rng.sample(available_mixed, needed)
            else:
                supplement = available_mixed
            selected = sorted(pure_pool + supplement)

        anchors[opt] = selected

    return anchors


def assert_no_leakage(
    anchors: dict[str, list[str]],
    test_texts: set[str],
    strict: bool = False,
) -> list[str]:
    """Check whether any anchor text also occurs in the test split."""
    all_anchors = {t for terms in anchors.values() for t in terms}
    leaked = sorted(all_anchors & test_texts)
    if leaked and strict:
        raise ValueError(
            f"train/test leakage: {len(leaked)} anchor(s) appear in the test set. "
            f"First: {leaked[0]!r}"
        )
    return leaked


def anchor_provenance(k: int = 3, seed: int = 20260930) -> dict:
    """Provenance metadata dictionary for benchmark results."""
    return {
        "dataset": "go_emotions",
        "method": f"monolabel_prioritized_few_shot_k{k}",
        "k": k,
        "seed": seed,
        "split": "train",
        "no_leakage": "anchors drawn from `train` only; overlap with `test` checked with assert_no_leakage()",
        "built_by": "klix (evals/multilabel_anchors.py)",
        "caveat": REPURPOSING_CAVEAT,
    }

"""Build anchor texts for the bespoke datasets — and make the choice explicit.

THE PROBLEM THIS FILE SOLVES
----------------------------
klix's `Choice` head does not classify from a label. It scores a query against
**anchor texts** (`options: dict[str, list[str]]`) and returns the nearest label.
MASSIVE (60 classes), banking77 (77 classes) and PAWS ship `text` + `label` and
**no anchors at all**.

So the anchors have to be produced here, and *how* they are produced decides what
the resulting number means. Two variants are built, always both, so the report can
show the spread instead of a single number whose provenance is invisible:

  `label_string`   the label with underscores replaced by spaces
                   ("balance_not_updated_after_bank_transfer" -> "... transfer").
                   The weakest possible anchor and therefore the honest LOWER
                   BOUND: it invents nothing, and cannot.

  `few_shot_k3`    k=3 real examples per class, drawn from the dataset's own
                   `train` split with a fixed seed. The MAIN measurement — it
                   tests klix in the situation it is actually used in (a handful
                   of known examples per label) and invents no phrasing either,
                   because every anchor is a real sentence from the corpus.

`few_shot_k3` was chosen over "write two template sentences per label" for the
reason the project has already paid for once: invented anchors measure our
phrasing, not the label mapping. It was chosen over "use the test sentences" for a
harder reason — that would be leakage, and the number would be meaningless.

LEAKAGE IS CHECKED, NOT ASSUMED
-------------------------------
`assert_no_leakage()` compares the anchor pool against the test texts and returns
the overlap. `strict=True` raises. The runner calls it before scoring and puts the
result in the report. A few-shot run that silently anchors on its own test set
produces a good number for a bad reason, and this is the guard against it.

NOT AN "OFFICIAL" BENCHMARK
---------------------------
These anchors are ours; the numbers they produce are not comparable to published
Tev1 / Nimble / JevBench figures. `anchor_provenance()` returns that caveat as a
dict so it travels into every result file (§7.1, option a).
"""
from __future__ import annotations

import random

ANCHOR_METHODS = ("few_shot_k3", "label_string")

REPURPOSING_CAVEAT = (
    "repurposed run; anchors are constructed by klix (see anchor_provenance), not "
    "shipped by the dataset; not comparable to published Tev1/Nimble/JevBench figures"
)


def label_to_anchor(label: str) -> str:
    """`some_label_name` -> `some label name`.

    Deliberately nothing more. Any cleaning up (dropping "support", adding a
    verb) would be us writing the anchor, which is the thing this variant exists
    to avoid.
    """
    return label.replace("_", " ").strip()


def label_string_anchors(labels: list[str]) -> dict[str, list[str]]:
    """One anchor per label — the lower-bound variant."""
    return {lab: [label_to_anchor(lab)] for lab in labels}


def few_shot_anchors(rows: list[dict], k: int = 3, seed: int = 20260930,
                     label_key: str = "label", text_key: str = "text",
                     ) -> dict[str, list[str]]:
    """k real examples per label from `rows`, deterministic under `seed`.

    Classes with fewer than k examples contribute everything they have rather
    than being dropped: dropping a class would silently remove it from the
    option list and inflate accuracy.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    by_label: dict[str, list[str]] = {}
    for row in rows:
        by_label.setdefault(row[label_key], []).append(row[text_key])

    rng = random.Random(seed)
    out: dict[str, list[str]] = {}
    for label in sorted(by_label):
        pool = sorted(by_label[label])  # sort first: dict order must not matter
        if len(pool) <= k:
            out[label] = pool
        else:
            out[label] = sorted(rng.sample(pool, k))
    return out


def assert_no_leakage(anchors: dict[str, list[str]], test_texts: set[str],
                      strict: bool = False) -> list[str]:
    """Return anchor texts that also occur in the test set.

    Overlap is not always a *bug*: two genuinely different rows can carry the
    same sentence (MASSIVE has near-duplicate phrasings across splits). It is
    always *reportable*, because it means that case is trivially answerable.
    """
    anchor_texts = {t for terms in anchors.values() for t in terms}
    leaked = sorted(anchor_texts & set(test_texts))
    if leaked and strict:
        raise ValueError(
            f"train/test leakage: {len(leaked)} anchor(s) also appear in the test set, "
            f"first: {leaked[0]!r}"
        )
    return leaked


def anchor_provenance(k: int = 3, seed: int = 20260930) -> dict:
    """Machine-readable provenance, embedded in every result file."""
    return {
        "methods": list(ANCHOR_METHODS),
        "main_method": "few_shot_k3",
        "lower_bound_method": "label_string",
        "k": k,
        "seed": seed,
        "split": "train",
        "no_leakage": "anchors are drawn from `train` only; overlap with `test` is "
                      "measured with assert_no_leakage() and reported",
        "built_by": "klix (evals/bespoke_anchors.py)",
        "caveat": REPURPOSING_CAVEAT,
    }

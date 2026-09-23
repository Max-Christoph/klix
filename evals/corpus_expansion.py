"""E.1 corpus expansion: programmatic paraphrase variants for the labeled sets.

The existing 60 labeled cases stay frozen. This module DERIVES additional
labeled cases from the anchor schemas via systematic templates:
- casing + punctuation variation
- filler/prefix wrapping ("kurze frage: ...", "could you help me with ...")
- urgency/politeness suffixes ("asap", "bitte", "(dringend)")

Every generated case carries (text, expected_label) with the label inherited
from the anchor class it was built from — no human labeling needed, and the
original 60 cases are never touched.

IMPORTANT honesty note: variants are template paraphrases of the anchors
themselves. They exercise wording robustness, NOT the hard adversarial cases
of the frozen 60. Variant accuracy is systematically higher; use the expanded
corpus to compare configs against EACH OTHER, not as an absolute benchmark.
"""

from __future__ import annotations

import random

_EN_PREFIXES = ["", "quick question: ", "could you help me with ", ""]
_DE_PREFIXES = ["", "kurze frage: ", "kannst du mir helfen bei ", ""]
_SUFFIXES = ["", " asap", " bitte", " - danke", ""]


def generate_variants(anchors: dict[str, list[str]], per_anchor: int = 3,
                      seed: int = 42) -> list[tuple[str, str]]:
    """Generates (text, label) variants from an anchor schema (deterministic)."""
    rng = random.Random(seed)
    out: list[tuple[str, str]] = []
    for label, examples in anchors.items():
        for anchor in examples:
            for k in range(per_anchor):
                kind = k % 3
                if kind == 0:
                    text = anchor.capitalize() + ("?" if rng.random() < 0.4 else ".")
                elif kind == 1:
                    pre = rng.choice(_EN_PREFIXES + _DE_PREFIXES)
                    text = (pre + anchor) if pre else anchor
                else:
                    suffix = rng.choice(_SUFFIXES)
                    text = anchor + suffix if suffix else f"bitte {anchor} (dringend)"
                out.append((text, label))
    return out


def build_expanded_corpus(sets: list[tuple[str, dict, list]]) -> dict:
    """Expands each domain's case list with generated variants.

    Duplicates against the original cases are filtered by exact text match.
    Returns {"domains": {name: {"options", "base", "variants", "cases"}},
             "total": int}.
    """
    result: dict = {"domains": {}, "total": 0}
    for name, options, tests in sets:
        seen = {t for t, _ in tests}
        variants = [(t, lab) for t, lab in generate_variants(options, per_anchor=3)
                    if t not in seen]
        result["domains"][name] = {
            "options": options,
            "base": list(tests),
            "variants": variants,
            "cases": list(tests) + variants,
        }
        result["total"] += len(tests) + len(variants)
    return result
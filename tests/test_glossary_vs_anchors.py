"""Structural facts about the glossary's reach, pinned. NOT the ablation outcome.

An earlier version of this file asserted the ablation's *result* — that bilingual
anchors were not clearly better than the glossary+bridge. That was a mistake: at
n=20 the 95 % CI spans −10…+20 points, so nothing about (b) vs (c) is established,
and pinning an unresolvable difference as if it were a finding would have frozen a
non-result into a regression gate.

What IS established, and asserted here, is the mechanism — it is a fact about the
code path, not a statistical estimate:

  1. Under `classifier="linear"` the sparse channel cannot influence the decision:
     the probe predicts from the dense vector and returns before the glossary
     expansion runs (`heads.py:1170-1171`, return at `:1197`). Measured by sweeping
     `glossary_weight` 0 → 50: exactly one answer pattern.
  2. Under `classifier="centroid"` the channel IS in the decision path: the same
     sweep yields more than one answer pattern on this schema.
  3. The glossary does fire (it expands German queries), so a flat result in (1)
     is not "the glossary never ran".

If (1) ever stops holding — because the linear path is changed to consume the
sparse channel — the README's explanation of the `linear` column becomes stale and
this test fails to say so. That is its purpose.

The ablation's numbers live in evals/glossary_vs_anchors_result.json and
docs/glossary-vs-anchors.md; they are deliberately NOT asserted here, because a
20-case count is not stable enough to gate on.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

from klix import Choice, DecisionEngine  # noqa: E402
from klix.glossaries import curated  # noqa: E402

# Single source of truth for the anchor/test sets: the eval script itself.
_spec = importlib.util.spec_from_file_location(
    "abl", REPO / "evals" / "glossary_vs_bilingual_anchors.py")
abl = importlib.util.module_from_spec(_spec)
sys.modules["abl"] = abl
_spec.loader.exec_module(abl)

WEIGHTS = (0.0, 0.5, 5.0, 50.0)


def answer_pattern(classifier, weight):
    eng = DecisionEngine(glossary=curated())
    eng.add_head(Choice(name="c", options=abl.EN_ONLY, classifier=classifier,
                        glossary_weight=weight))
    eng.compile()
    return tuple(eng.decide(q).c for q, _ in abl.TEST_DE)


def patterns(classifier):
    return {answer_pattern(classifier, w) for w in WEIGHTS}


class TestSparseChannelReach:
    def test_linear_never_consults_the_sparse_channel(self):
        """The finding the README now explains — verified by weight sweep.

        `glossary_weight` from 0 to 50 changes nothing under `linear`, while the
        same sweep does change answers under `centroid`. If this ever flips, the
        linear path started consuming the sparse channel and the README's
        explanation of that column must be rewritten.
        """
        got = patterns("linear")
        assert len(got) == 1, (
            f"'linear' now yields {len(got)} answer patterns over glossary_weight "
            f"{WEIGHTS} — the sparse channel has become reachable on that path, so "
            f"README.md and docs/glossary-vs-anchors.md must be revisited")

    def test_centroid_does_consult_it(self):
        """The complement: the channel is reachable where this project claims it is."""
        got = patterns("centroid")
        assert len(got) > 1, (
            f"'centroid' yields only {len(got)} answer pattern(s) over glossary_weight "
            f"{WEIGHTS} — the sparse channel may have left the centroid decision path, "
            f"which would invalidate the (a)/(b)/(c)/(d) reading of the ablation")


class TestTheGlossaryActuallyFires:
    def test_it_expands_a_majority_of_german_queries(self):
        """Otherwise a flat result would just mean 'nothing matched'."""
        eng = DecisionEngine(glossary=curated())
        eng.add_head(Choice(name="c", options=abl.EN_ONLY, classifier="centroid"))
        eng.compile()
        fired = sum(1 for q, _ in abl.TEST_DE if eng.heads[0].expand_query_terms(q))
        assert fired >= 5, (
            f"only {fired}/10 German queries expanded — the preset no longer covers "
            f"this domain, so any flat result has a different cause than documented")


class TestTheBridgeDoesNotHurt:
    def test_german_accuracy_does_not_drop_when_the_glossary_is_added(self):
        """Weaker than the claim it replaces, and deliberately so: this only
        excludes 'the bridge actively breaks German routing', which is checkable
        at n=20. It says nothing about (b) vs (c)."""
        without = sum(1 for t, e in abl.TEST_DE
                      if _decide(abl.EN_ONLY, False, "centroid", t) == e)
        with_g = sum(1 for t, e in abl.TEST_DE
                     if _decide(abl.EN_ONLY, True, "centroid", t) == e)
        assert with_g >= without, (
            f"glossary lowered German accuracy: without={without} with={with_g}")


def _decide(anchors, use_glossary, classifier, text):
    eng = DecisionEngine(glossary=curated() if use_glossary else None)
    eng.add_head(Choice(name="c", options=anchors, classifier=classifier))
    eng.compile()
    return eng.decide(text).c

"""The glossary-vs-bilingual-anchors ablation, pinned.

`evals/glossary_vs_bilingual_anchors.py` measures whether a glossary replaces
bilingual anchors or adds to them. Its outcome drove a README correction, so the
claims the README now makes are asserted here: if a future change moves the
numbers, this fails and the documentation gets revisited instead of silently
going stale.

What is pinned is the STRUCTURE of the finding, not the exact counts:

  1. The glossary fires on the monolingual-EN schema (queries get expanded) — so
     the null result in (d) is not "the glossary never ran".
  2. Under `linear`, the glossary does not change a single German answer on that
     schema. That is the observation that makes "anchors already cover the
     language" the operative explanation.
  3. (b) is not worse than (a) on the German test cases — the bridge does not hurt.
  4. Cell (d) does not beat cell (c). If it ever clearly does, the README's
     "bilingual anchors are equally good" wording must be rewritten.

The exact per-cell scores live in evals/glossary_vs_anchors_result.json and in
docs/glossary-vs-anchors.md. Asserting them here would make the test fail on any
minor model or dependency change, which would be noise rather than signal — the
numbers belong in the eval script's output, the direction belongs in the test.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

from klix import Choice, DecisionEngine  # noqa: E402
from klix.glossaries import curated  # noqa: E402

# Load the ablation module for its anchor and test sets (single source of truth —
# the test must run against exactly the data the eval measures).
_spec = importlib.util.spec_from_file_location(
    "abl", REPO / "evals" / "glossary_vs_bilingual_anchors.py")
abl = importlib.util.module_from_spec(_spec)
sys.modules["abl"] = abl
_spec.loader.exec_module(abl)


def decide(anchors, use_glossary, classifier, text):
    eng = DecisionEngine(glossary=curated() if use_glossary else None)
    eng.add_head(Choice(name="c", options=anchors, classifier=classifier))
    eng.compile()
    return eng.decide(text)


def expansions(anchors, text):
    eng = DecisionEngine(glossary=curated())
    eng.add_head(Choice(name="c", options=anchors, classifier="centroid"))
    eng.compile()
    return eng.heads[0].expand_query_terms(text)


class TestTheGlossaryActuallyFires:
    def test_it_expands_a_majority_of_german_queries(self):
        """Otherwise the (d) null result would just mean 'the glossary never ran'."""
        fired = sum(1 for q, _ in abl.TEST_DE if expansions(abl.EN_ONLY, q))
        assert fired >= 5, (
            f"only {fired}/10 German queries expanded — the preset no longer "
            f"covers this domain, so the ablation's null result for (d) has a "
            f"different cause than documented")

    def test_a_known_query_is_bridged_to_its_concept(self):
        """elternzeit -> parental_leave is the one case the glossary visibly fixes."""
        assert "parental_leave" in curated().match_concepts("meine elternzeit")
        got = decide(abl.EN_ONLY, True, "centroid", "ich moechte meine elternzeit verlaengern")
        assert got.c == "hr"


class TestDirectionOfTheFinding:
    def test_linear_german_answers_are_unaffected_by_the_glossary(self):
        """The observation behind 'the anchors already carry the language'.

        If a future change makes the glossary move linear answers on this schema,
        that is not necessarily bad — but it invalidates the explanation the docs
        give, so the docs need revisiting. Failing here is the prompt for that.
        """
        changed = []
        for text, _ in abl.TEST_DE:
            without = decide(abl.EN_ONLY, False, "linear", text).c
            with_g = decide(abl.EN_ONLY, True, "linear", text).c
            if without != with_g:
                changed.append((text, without, with_g))
        assert not changed, (
            "the glossary now moves linear answers on this schema: "
            f"{changed} — docs/glossary-vs-anchors.md and the README explanation "
            f"must be revisited")

    def test_the_bridge_does_not_hurt_german_accuracy(self):
        """(b) >= (a) on German: expanding must not cost correct answers."""
        a = sum(1 for t, e in abl.TEST_DE if decide(abl.EN_ONLY, False, "centroid", t).c == e)
        b = sum(1 for t, e in abl.TEST_DE if decide(abl.EN_ONLY, True, "centroid", t).c == e)
        assert b >= a, f"glossary lowered German accuracy: (a)={a} (b)={b}"

    def test_bilingual_anchors_are_not_clearly_better_than_the_glossary(self):
        """(c) must not beat (b) by a margin — that is what the README claims.

        A large, stable gap here would mean bilingual anchors are strictly the
        better answer and the README's 'equally good' wording is too generous.
        """
        b = sum(1 for t, e in abl.TEST_DE if decide(abl.EN_ONLY, True, "centroid", t).c == e)
        c = sum(1 for t, e in abl.TEST_DE if decide(abl.BILINGUAL, False, "centroid", t).c == e)
        assert c - b <= 2, (
            f"(c) beats (b) by {c - b} of {len(abl.TEST_DE)} — bilingual anchors now "
            f"look strictly better; the README wording needs correcting")

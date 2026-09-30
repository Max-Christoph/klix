"""Unit tests for the bespoke anchor construction (§7.2).

The point of these tests is not that the code runs — it is that the two anchor
variants stay distinguishable and that the **no-leakage** property is enforced
rather than assumed. A few-shot run that accidentally uses a test sentence as an
anchor produces a good number for a bad reason, and that is exactly the failure
this file exists to make impossible.
"""
from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from evals.bespoke_anchors import (  # noqa: E402
    ANCHOR_METHODS,
    assert_no_leakage,
    anchor_provenance,
    few_shot_anchors,
    label_to_anchor,
)


class TestLabelStringVariant:
    def test_replaces_underscores_with_spaces(self):
        assert label_to_anchor("atm_support") == "atm support"

    def test_is_a_collapse_not_a_sentence(self):
        # The whole reason this variant is the *lower bound*: it cannot invent
        # phrasing, so a long label stays a stiff noun phrase.
        out = label_to_anchor("balance_not_updated_after_bank_transfer")
        assert out == "balance not updated after bank transfer"
        assert "_" not in out

    def test_is_deterministic(self):
        first = label_to_anchor("apple_pay_or_google_pay")
        assert first == label_to_anchor("apple_pay_or_google_pay")

    def test_empty_label_does_not_crash(self):
        assert label_to_anchor("") == ""


class TestFewShotVariant:
    ROWS = [
        {"label": "alarm_set", "text": "weck mich um sieben"},
        {"label": "alarm_set", "text": "stell den wecker auf neun"},
        {"label": "alarm_set", "text": "wecke mich morgen frueh"},
        {"label": "alarm_set", "text": "alarm um sechs uhr"},
        {"label": "alarm_query", "text": "wann klingelt mein wecker"},
        {"label": "alarm_query", "text": "welcher alarm ist gestellt"},
        {"label": "alarm_query", "text": "habe ich einen alarm"},
    ]

    def test_takes_k_examples_per_label(self):
        out = few_shot_anchors(self.ROWS, k=3)
        assert set(out) == {"alarm_set", "alarm_query"}
        assert all(len(v) == 3 for v in out.values())

    def test_is_deterministic_across_calls(self):
        a = few_shot_anchors(self.ROWS, k=3)
        b = few_shot_anchors(self.ROWS, k=3)
        assert a == b

    def test_different_seed_can_change_the_selection(self):
        # Not required to differ for every input, but the seed must actually be
        # wired up: with 4 candidates and k=3 there are 4 possible draws.
        picks = {
            tuple(few_shot_anchors(self.ROWS, k=3, seed=s)["alarm_set"])
            for s in range(12)
        }
        assert len(picks) > 1, "seed has no effect — k=3 selection is fixed"

    def test_anchors_come_from_the_data(self):
        out = few_shot_anchors(self.ROWS, k=3)
        pool = {r["text"] for r in self.ROWS}
        assert all(t in pool for terms in out.values() for t in terms)

    def test_label_with_fewer_than_k_examples_takes_what_exists(self):
        rows = [{"label": "a", "text": "one"}, {"label": "b", "text": "two"},
                {"label": "b", "text": "three"}, {"label": "b", "text": "four"}]
        out = few_shot_anchors(rows, k=3)
        assert len(out["a"]) == 1
        assert len(out["b"]) == 3

    def test_k_zero_is_rejected(self):
        with pytest.raises(ValueError, match="k must be >= 1"):
            few_shot_anchors(self.ROWS, k=0)


class TestLeakageGuard:
    def test_detects_a_leaked_anchor(self):
        anchors = {"alarm_set": ["weck mich um sieben", "LEAKED TEST TEXT"]}
        leaked = assert_no_leakage(anchors, test_texts={"LEAKED TEST TEXT", "other"})
        assert leaked == ["LEAKED TEST TEXT"]

    def test_passes_when_disjoint(self):
        anchors = {"alarm_set": ["a", "b"]}
        assert assert_no_leakage(anchors, test_texts={"c", "d"}) == []

    def test_raises_in_strict_mode(self):
        with pytest.raises(ValueError, match="train/test leakage"):
            assert_no_leakage({"x": ["same"]}, test_texts={"same"}, strict=True)


class TestProvenance:
    def test_provenance_names_both_methods(self):
        prov = anchor_provenance()
        assert set(prov["methods"]) == set(ANCHOR_METHODS)
        assert prov["caveat"], "provenance must carry the repurposing caveat"

    def test_provenance_states_the_k_and_the_split(self):
        prov = anchor_provenance(k=3)
        assert prov["k"] == 3
        assert prov["split"] == "train"
        assert "test" in prov["no_leakage"]

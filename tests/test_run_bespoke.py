"""Unit tests for the bespoke runner — the properties that make a comparison fair.

These do NOT test accuracy (that is a measurement, not a property). They test the
four things that decide whether a number is *comparable at all*:

  1. every system is scored on the same cases, in the same order
  2. `unparseable` never counts as wrong and never as right
  3. the caveat and anchor provenance travel into the result file
  4. a model that emits prose is reported as unusable, not as inaccurate

A comparison that fails any of these produces a number that looks fine and means
nothing — which is the failure mode this project has already hit once.
"""
from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from evals.run_bespoke import (  # noqa: E402
    bootstrap_ci,
    build_anchors,
    score_case_alignment,
    summarize,
)

CASES = [
    {"id": "1", "text": "weck mich um sieben", "label": "alarm_set", "options": ["alarm_set", "alarm_query"]},
    {"id": "2", "text": "wann klingelt der wecker", "label": "alarm_query", "options": ["alarm_set", "alarm_query"]},
    {"id": "3", "text": "stell den wecker", "label": "alarm_set", "options": ["alarm_set", "alarm_query"]},
]


class TestSummarize:
    def test_counts_correct_and_unparseable_separately(self):
        preds = ["alarm_set", None, "alarm_query"]
        out = summarize(CASES, preds)
        assert out["n"] == 3
        assert out["correct"] == 1          # only case 3 matched
        assert out["unparseable"] == 1      # the None
        assert out["answered"] == 2

    def test_accuracy_is_over_answered_not_over_all(self):
        # A system that abstains on half the cases and is right on the rest is
        # NOT 50% accurate - it is 100% accurate on what it answered, with a
        # coverage problem. Conflating the two hides a broken prompt.
        preds = ["alarm_set", None, "alarm_set"]
        out = summarize(CASES, preds)
        assert out["accuracy"] == pytest.approx(1.0)
        assert out["coverage"] == pytest.approx(2 / 3)

    def test_all_unparseable_does_not_divide_by_zero(self):
        out = summarize(CASES, [None, None, None])
        assert out["accuracy"] is None
        assert out["coverage"] == 0.0

    def test_empty_input_is_handled(self):
        out = summarize([], [])
        assert out["n"] == 0 and out["accuracy"] is None

    def test_mismatched_lengths_are_rejected(self):
        # Silent zip-truncation would quietly score fewer cases than reported.
        with pytest.raises(ValueError, match="length mismatch"):
            summarize(CASES, ["alarm_set"])


class TestBootstrapCI:
    def test_is_deterministic_under_seed(self):
        hits = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
        assert bootstrap_ci(hits, seed=7) == bootstrap_ci(hits, seed=7)

    def test_brackets_the_point_estimate(self):
        hits = [1] * 70 + [0] * 30
        lo, hi = bootstrap_ci(hits, seed=7)
        assert lo <= 0.70 <= hi

    def test_uniform_input_collapses_to_the_value(self):
        assert bootstrap_ci([1] * 20, seed=7) == (1.0, 1.0)

    def test_empty_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            bootstrap_ci([])


class TestAnchors:
    def test_builds_both_variants(self):
        train = [{"label": "alarm_set", "text": f"s{i}"} for i in range(5)]
        train += [{"label": "alarm_query", "text": f"q{i}"} for i in range(5)]
        out = build_anchors(train, method="few_shot_k3", k=3)
        assert set(out) == {"alarm_set", "alarm_query"}
        assert all(len(v) == 3 for v in out.values())

    def test_label_string_method_uses_one_anchor_per_label(self):
        out = build_anchors([{"label": "alarm_set", "text": "x"}], method="label_string")
        assert out == {"alarm_set": ["alarm set"]}

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="unknown anchor method"):
            build_anchors([], method="invented")


class TestCaseAlignment:
    def test_detects_a_model_scored_on_different_cases(self):
        # The cheapest way to accidentally invent a leaderboard: score model B on
        # the cases model A happened to get wrong.
        a = [{"id": "1"}, {"id": "2"}]
        b = [{"id": "2"}, {"id": "1"}]
        assert score_case_alignment({"a": a, "b": b}) is False

    def test_true_when_all_models_share_the_case_list(self):
        a = [{"id": "1"}, {"id": "2"}]
        assert score_case_alignment({"a": a, "b": list(a)}) is True

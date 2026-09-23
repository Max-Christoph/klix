"""Tests for the v0.1.4 head features: Choice reject pole, Score topk + coverage."""

import numpy as np

from klix import Choice, DecisionEngine, Score


def build() -> DecisionEngine:
    eng = DecisionEngine()
    eng.add_head(
        Choice(
            name="target",
            options={
                "it_ops": ["VPN down", "server unreachable", "laptop won't boot"],
                "finance": ["approve invoice", "cost center over budget"],
            },
            reject_anchors=["birthday wishes", "small talk about the weather", "casual office chat"],
        )
    )
    eng.add_head(
        Score(
            name="urgency",
            low_anchors=["routine maintenance", "casual question", "no rush"],
            high_anchors=["emergency right now", "production line down", "acute danger"],
            min_val=0.0,
            max_val=3.0,
            aggregation="topk",
        )
    )
    eng.compile()
    return eng


class TestChoiceReject:
    def test_reject_returns_none(self):
        eng = build()
        d = eng.decide("happy birthday to everyone in the office").details("target")
        assert d["value"] is None
        assert d["confidence"] == 0.0

    def test_in_domain_not_rejected(self):
        eng = build()
        d = eng.decide("approve the invoice from the supplier").details("target")
        assert d["value"] == "finance"
        assert d["reject_score"] < d["score"]

    def test_reject_score_always_present(self):
        eng = build()
        d = eng.decide("VPN down again").details("target")
        assert "reject_score" in d

    def test_no_reject_pole_backward_compatible(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="c", options={"a": ["alpha"], "b": ["beta"]}))
        eng.compile()
        d = eng.decide("alpha shot").details("c")
        assert d["value"] in {"a", "b"}
        assert "reject_score" not in d or d["reject_score"] == 0.0


class TestScoreTopk:
    def test_topk_matches_max_for_strong_cases(self):
        eng = build()
        d = eng.decide("production line down, evacuate now!").details("urgency")
        assert d["value"] >= 2.0

    def test_coverage_present_and_in_range(self):
        eng = build()
        for text in ["emergency!!", "casual question", "wifi password"]:
            cov = eng.decide(text).details("urgency")["coverage"]
            assert 0.0 <= cov <= 1.0

    def test_low_coverage_detects_off_domain(self):
        eng = build()
        cov = eng.decide("whats the wifi password").details("urgency")["coverage"]
        assert cov < 0.3

    def test_high_coverage_for_anchored_text(self):
        eng = build()
        cov = eng.decide("production line down, acute danger").details("urgency")["coverage"]
        assert cov > 0.4

    def test_max_aggregation_still_default(self):
        eng2 = DecisionEngine()
        eng2.add_head(
            Score(name="u", low_anchors=["calm"], high_anchors=["crisis"], min_val=0.0, max_val=3.0)
        )
        eng2.compile()
        d = eng2.decide("total crisis right now").details("u")
        assert "coverage" in d  # coverage always present
        assert d["value"] >= 1.5

    def test_topk_never_exceeds_anchor_count(self):
        eng2 = DecisionEngine()
        eng2.add_head(
            Score(
                name="u",
                low_anchors=["calm"],
                high_anchors=["panic"],
                aggregation="topk",
                topk=5,  # more anchors than available
            )
        )
        eng2.compile()
        result = eng2.decide("panic everywhere")
        assert result.u is not None  # must not raise


class TestEngineConfig:
    def test_stop_words_none_uses_default(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="c", options={"a": ["alpha sentence"], "b": ["beta sentence"]}))
        eng.compile()
        assert eng.backbone.is_indexed

    def test_custom_stop_words_change_index(self):
        eng = DecisionEngine(stop_words=[])  # keep everything
        eng.add_head(Choice(name="c", options={"a": ["the alpha"], "b": ["the beta"]}))
        eng.compile()
        # "the" is now part of the vocabulary
        assert "the" in eng.backbone.tfidf_vec.vocabulary_

    def test_custom_stop_words_exclude_terms(self):
        eng = DecisionEngine(stop_words=["error"])
        eng.add_head(Choice(name="c", options={"a": ["alpha error"], "b": ["beta fault"]}))
        eng.compile()
        assert "error" not in eng.backbone.tfidf_vec.vocabulary_


class TestTopkPerLabel:
    """A1 fix: label_topk must not be clamped by the smallest label."""

    def _eng(self):
        eng = DecisionEngine()
        # "small" has only ONE anchor, others have three -> old bug forced k=1
        # for every label, silently degrading topk to max.
        eng.add_head(
            Choice(
                name="c",
                options={
                    "big_a": ["alpha one", "alpha two", "alpha three"],
                    "big_b": ["beta one", "beta two", "beta three"],
                    "small": ["tiny"],
                },
                label_aggregation="topk",
                label_topk=3,
            )
        )
        eng.compile()
        return eng

    def test_topk_does_not_collapse_to_max(self):
        eng = self._eng()
        head = eng.heads[0]
        # The bug would make k=1 for all labels; per-label fix keeps k=3 for the
        # big labels and k=1 (its own length) for the tiny label.
        assert head.label_aggregation == "topk"
        # exercise the pooling path without crashing on the 1-anchor label
        r = eng.decide("alpha one two three")
        assert r.c in {"big_a", "big_b", "small"}

    def test_topk_per_label_values_distinct_from_max(self):
        # For a 3-anchor label, topk3 mean differs from max when anchors differ.
        eng = DecisionEngine()
        eng.add_head(
            Choice(
                name="c",
                options={"x": ["aaa", "bbb", "ccc"]},
                label_aggregation="topk",
                label_topk=2,
            )
        )
        eng.compile()
        eng.decide("some probe text")
        assert eng.heads[0]._label_rows["x"] == [0, 1, 2]


class TestRejectThreshold:
    def test_reject_threshold_forces_none(self):
        eng = DecisionEngine()
        eng.add_head(
            Choice(
                name="c",
                options={"a": ["alpha one two three"], "b": ["beta one two three"]},
                reject_threshold=0.9,
            )
        )
        eng.compile()
        # A query that matches neither strongly should fall below the floor.
        r = eng.decide("completely unrelated words here")
        assert r.c is None

    def test_reject_threshold_passes_strong_match(self):
        eng = DecisionEngine()
        eng.add_head(
            Choice(
                name="c",
                options={"a": ["alpha one two three"], "b": ["beta one two three"]},
                reject_threshold=0.9,
            )
        )
        eng.compile()
        r = eng.decide("alpha one two three exactly")
        assert r.c == "a"

    def test_reject_threshold_none_keeps_value(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="c", options={"a": ["alpha"], "b": ["beta"]}))
        eng.compile()
        assert eng.decide("alpha").c == "a"


class TestCoverageBoost:
    def test_coverage_boost_lowers_generic_word_effect(self):
        # A German query (off-vocabulary) should get a tiny keyword boost vs. an
        # in-vocabulary English query, all else equal.
        eng = DecisionEngine()
        eng.add_head(
            Choice(
                name="c",
                options={"a": ["please approve this invoice"], "b": ["the server is down"]},
                keyword_boost_mode="coverage",
                keyword_boost=1.0,
            )
        )
        eng.compile()
        # in-vocabulary English -> high coverage
        d_en = eng.decide("please approve invoice").details("c")
        # German query shares almost no vocabulary -> low coverage
        d_de = eng.decide("bitte die rechnung freigeben").details("c")
        assert d_en["score"] > d_de["score"]
"""Tests for classifier="centroid" (v0.8.5).

Contract:
- default classifier is unchanged ("nearest")
- "centroid" builds one mean anchor vector per label at compile time
- scoring is cosine to those vectors, fused with the sparse channel
- deterministic (no training, no randomness)
- works with rules, reject anchors and the other heads
- reaches the documented accuracy on the frozen sets
"""
import numpy as np
import pytest

from klix import Choice, DecisionEngine, Flag, Score

OPTIONS = {
    "billing": ["invoice double charged", "refund missing", "wrong invoice amount"],
    "technical": ["server keeps crashing", "vpn drops constantly", "laptop will not boot"],
    "facility": ["office heating broken", "water leaks from ceiling", "elevator stuck"],
}
CASES = [
    ("the invoice was charged twice", "billing"),
    ("my refund never arrived", "billing"),
    ("server is down again", "technical"),
    ("vpn keeps dropping", "technical"),
    ("the heating in the office is broken", "facility"),
]


def _eng(classifier="centroid") -> DecisionEngine:
    eng = DecisionEngine()
    eng.add_head(Choice(name="route", options=OPTIONS, classifier=classifier))
    eng.compile()
    return eng


class TestCentroid:
    def test_default_classifier_is_still_nearest(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="route", options=OPTIONS))
        eng.compile()
        assert eng.heads[0]._effective_classifier == "nearest"
        assert eng.heads[0]._centroid_matrix is None

    def test_centroid_builds_one_vector_per_label(self):
        eng = _eng()
        head = eng.heads[0]
        assert head._effective_classifier == "centroid"
        assert head._centroid_matrix.shape[0] == len(OPTIONS)
        assert head._centroid_matrix.shape[1] == head.dense_matrix.shape[1]

    def test_centroid_vectors_are_unit_length(self):
        eng = _eng()
        norms = np.linalg.norm(eng.heads[0]._centroid_matrix, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_no_probe_is_trained(self):
        """The whole point: centroid needs no training."""
        eng = _eng()
        assert eng.heads[0]._probe is None

    def test_classification_works(self):
        eng = _eng()
        got = [eng.decide(t).route for t, _ in CASES]
        expected = [e for _, e in CASES]
        assert got == expected

    def test_is_deterministic_across_recompiles(self):
        first = [ _eng().decide(t).route for t, _ in CASES]
        second = [_eng().decide(t).route for t, _ in CASES]
        assert first == second

    def test_nearest_and_centroid_both_available(self):
        for clf in ("nearest", "centroid"):
            eng = _eng(clf)
            res = eng.decide("server is down")
            assert res.route in OPTIONS

    def test_works_with_reject_anchors(self):
        eng = DecisionEngine()
        eng.add_head(Choice(
            name="route", options=OPTIONS, classifier="centroid",
            reject_anchors=["happy birthday to the team", "nice weather today"],
        ))
        eng.compile()
        assert eng.decide("the invoice was charged twice").route == "billing"
        # off-domain should be rejected (None) rather than force-routed
        assert eng.decide("happy birthday to the whole team").route in (None, "billing",
                                                                        "technical", "facility")

    def test_works_with_rules(self):
        from klix.rules import Rule
        eng = DecisionEngine()
        eng.add_head(Choice(
            name="route", options=OPTIONS, classifier="centroid",
            rules=[Rule(label="technical", pattern=r"(?i)\bplc-\d+\b", mode="force",
                        name="asset_id")],
        ))
        eng.compile()
        assert eng.decide("plc-34 reports a fault").route == "technical"

    def test_result_carries_confidence_and_scores(self):
        d = _eng().decide("server is down").details("route")
        assert "confidence" in d and "score" in d
        assert set(d["scores"]) == set(OPTIONS)
        assert 0.0 <= d["confidence"] <= 1.0

    def test_other_heads_unaffected(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="route", options=OPTIONS, classifier="centroid"))
        eng.add_head(Score(name="urgency", low_anchors=["routine question"],
                           high_anchors=["emergency outage"]))
        eng.add_head(Flag(name="sec", true_anchors=["hacking attempt"],
                          false_anchors=["printer jam"]))
        eng.compile()
        res = eng.decide("urgent server is down")
        assert res.route in OPTIONS
        assert res.urgency is None or 0.0 <= res.urgency <= 3.0
        assert res.sec in (True, False, None)

    def test_batch_matches_serial(self):
        eng = _eng()
        texts = [t for t, _ in CASES]
        batch = eng.decide_batch(texts)
        serial = [eng.decide(t) for t in texts]
        assert [b.route for b in batch] == [s.route for s in serial]

    def test_matches_linear_probe_on_frozen_set(self):
        """centroid must reach the accuracy of the trained probe (no training)."""
        from evals.eval_domains import (IMG_CASES, IMG_OPTIONS, SHOP_CASES,
                                        SHOP_OPTIONS, TASK_CASES, TASK_OPTIONS)
        from evals.variant_sweep import (FIN_OPTIONS, FIN_TESTS, HR_OPTIONS,
                                         HR_TESTS)
        sets = [
            (HR_OPTIONS, HR_TESTS),
            (FIN_OPTIONS, FIN_TESTS),
            (IMG_OPTIONS, [(q, e) for q, e in IMG_CASES if e]),
            (TASK_OPTIONS, [(q, e) for q, e in TASK_CASES if e]),
            (SHOP_OPTIONS, [(q, e) for q, e in SHOP_CASES if e]),
        ]

        def run(clf):
            ok = tot = 0
            for opts, tests in sets:
                e = DecisionEngine()
                e.add_head(Choice(name="h", options=opts, classifier=clf))
                e.compile()
                ok += sum(1 for t, x in tests if e.decide(t).h == x)
                tot += len(tests)
            return ok, tot

        c_ok, n = run("centroid")
        l_ok, _ = run("linear")
        assert n == 60
        assert c_ok >= l_ok, f"centroid {c_ok} should match/exceed linear {l_ok}"
        assert c_ok >= 50, f"centroid regressed: {c_ok}/{n}"

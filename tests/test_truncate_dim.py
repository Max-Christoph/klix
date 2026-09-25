"""Tests for truncate_dim (v0.8.4): MRL-style embedding truncation.

Contract pinned here:
- default (None) is byte-identical to previous behaviour (no truncation)
- truncate_dim slices EVERY dense vector: anchors and queries alike
- vectors stay L2-normalized (cosine semantics preserved)
- heads stay decoupled: truncation is a backbone-level transform
"""
import numpy as np

from klix import Choice, DecisionEngine, Flag, Score


def _engine(**kw) -> DecisionEngine:
    eng = DecisionEngine(**kw)
    eng.add_head(Choice(name="route", options={
        "it": ["server down", "vpn broken"],
        "ot": ["plc error", "robot stopped"],
    }))
    return eng


class TestTruncateDim:
    def test_default_is_none_and_keeps_full_dim(self):
        eng = _engine()
        eng.compile()
        assert eng.backbone.truncate_dim is None
        assert eng.decide("server down").details("route")["value"] is not None
        # anchor matrix keeps the model's native dimension
        assert eng.heads[0].dense_matrix.shape[1] == 384

    def test_truncated_anchor_matrix_has_requested_dim(self):
        eng = _engine(truncate_dim=64)
        eng.compile()
        assert eng.heads[0].dense_matrix.shape[1] == 64

    def test_query_vector_is_truncated_too(self):
        eng = _engine(truncate_dim=64)
        eng.compile()
        encoded = eng.backbone.encode("server down")
        assert encoded.dense_vec.shape == (64,)

    def test_vectors_stay_l2_normalized(self):
        eng = _engine(truncate_dim=64)
        eng.compile()
        encoded = eng.backbone.encode("plc error")
        assert abs(float(np.linalg.norm(encoded.dense_vec)) - 1.0) < 1e-5

    def test_truncate_dim_above_native_dim_is_a_noop_slice(self):
        eng = _engine(truncate_dim=10000)
        eng.compile()
        encoded = eng.backbone.encode("server down")
        assert encoded.dense_vec.shape == (384,)

    def test_decision_still_works_and_is_deterministic(self):
        eng = _engine(truncate_dim=64)
        eng.compile()
        first = eng.decide("plc error on the line").route
        second = eng.decide("plc error on the line").route
        assert first == second

    def test_batch_matches_serial_with_truncation(self):
        eng = _engine(truncate_dim=64)
        eng.compile()
        texts = ["server down", "plc error", "vpn broken"]
        batch = eng.decide_batch(texts)
        serial = [eng.decide(t) for t in texts]
        for b, s in zip(batch, serial):
            assert b.route == s.route
            assert abs(b.details("route")["score"] - s.details("route")["score"]) < 1e-6

    def test_all_three_heads_work_with_truncation(self):
        eng = DecisionEngine(truncate_dim=64)
        eng.add_head(Choice(name="route", options={"a": ["server down"], "b": ["plc error"]}))
        eng.add_head(Score(name="urgency", low_anchors=["routine"], high_anchors=["emergency"]))
        eng.add_head(Flag(name="sec", true_anchors=["hacking"], false_anchors=["printer jam"]))
        eng.compile()
        res = eng.decide("urgent plc error")
        assert res.route in ("a", "b")
        assert res.sec in (True, False, None)
        assert res.urgency is None or 0.0 <= res.urgency <= 3.0

    def test_repeated_compile_does_not_stack_wrappers(self):
        """Recompiling must not wrap embed() a second time (no double truncation)."""
        eng = _engine(truncate_dim=64)
        eng.compile()
        first = eng.heads[0].dense_matrix.shape[1]
        eng._compiled = False
        eng.compile()
        assert eng.heads[0].dense_matrix.shape[1] == first == 64

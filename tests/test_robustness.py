"""Tests for the 0.5.0 robustness wave: stopwords (Joghurt-Fehler), CV-based
calibration, and decide_batch."""

import numpy as np
import pytest

from klix import Choice, DecisionEngine, Flag, Score


class TestStopwordRobustness:
    """K1: grammatical fillers must not override semantics (Joghurt-Fehler)."""

    def _build(self):
        eng = DecisionEngine()
        eng.add_head(Choice(
            name="route",
            options={
                "security": [
                    "verdächtiger login aus fremdem land",
                    "ransomware hat den fileserver verschlüsselt",
                ],
                "not_relevant": [
                    "wer hat den letzten joghurt aus dem kühlschrank genommen",
                    "die räume müssten mal wieder geputzt werden",
                ],
            },
        ))
        eng.compile()
        return eng

    def test_fillers_filtered_from_vocabulary(self):
        eng = self._build()
        vocab = eng.backbone.tfidf_vec.vocabulary_
        for filler in ["den", "hat", "der", "die", "das", "ist"]:
            assert filler not in vocab, f"'{filler}' should be filtered from vocabulary"

    def test_admin_ticket_beats_joghurt_anchor(self):
        eng = self._build()
        d = eng.decide("jemand hat sich in den admin account eingeloggt, um 3 uhr").details("route")
        assert d["value"] == "security", f"filler words still override: {d}"

    def test_explicit_disable_still_possible(self):
        eng = DecisionEngine(stop_words=[])
        eng.add_head(Choice(name="c", options={"a": ["the alpha"], "b": ["the beta"]}))
        eng.compile()
        assert "the" in eng.backbone.tfidf_vec.vocabulary_


class TestCalibrateCrossValidation:
    """K2: threshold selection must be robust against small-sample overfit."""

    def _build(self):
        eng = DecisionEngine()
        eng.add_head(Flag(
            name="sec",
            true_anchors=["hacker attack", "ransomware infection", "compromised root login"],
            false_anchors=["printer jam", "hardware broken", "network outage"],
        ))
        eng.compile()
        return eng

    def test_cv_used_for_sufficient_samples(self):
        eng = self._build()
        samples = [
            ("ransomware encrypted our files", True),
            ("someone hacked the admin account", True),
            ("root login from a foreign country", True),
            ("phishing email to all staff", True),
            ("the printer is jammed again", False),
            ("the monitor flickers", False),
            ("wifi is slow in meeting room 1", False),
            ("hardware defect, please replace", False),
            ("cable mess under the desk", False),
            ("suspicious data exfiltration at night", True),
        ]
        report = eng.calibrate("sec", samples)
        assert report["cv"] is not None  # cv path taken
        assert report["cv"] >= 2
        assert report["spread"] is not None
        assert 0.0 <= report["threshold"] <= 1.0

    def test_fallback_for_tiny_samples(self):
        eng = self._build()
        report = eng.calibrate("sec", [("ransomware", True), ("printer jam", False)])
        assert report["cv"] is None
        assert report["warning"] is not None  # n < 8

    def test_report_contains_stability_info(self):
        eng = self._build()
        samples = [
            ("ransomware hit us", True), ("hacker attack now", True),
            ("phishing mail found", True), ("data breach last night", True),
            ("printer jam", False), ("monitor broken", False),
            ("wifi slow", False), ("mouse cable snapped", False),
        ]
        report = eng.calibrate("sec", samples)
        assert "cv" in report and "spread" in report
        # clean separation -> folds should agree (low spread)
        assert report["spread"] is None or report["spread"] <= 0.5


class TestDecideBatch:
    """K4: bulk processing must be faster per item and consistent with decide()."""

    def _build(self):
        eng = DecisionEngine()
        eng.add_head(Choice(
            name="route",
            options={
                "billing": ["approve invoice", "cost center over budget"],
                "technical": ["server down", "vpn keeps dropping"],
            },
        ))
        eng.add_head(Score(
            name="urgency",
            low_anchors=["routine maintenance", "casual question"],
            high_anchors=["emergency right now", "production line down"],
            min_val=0.0, max_val=3.0,
        ))
        eng.compile()
        return eng

    def test_batch_matches_single_results(self):
        eng = self._build()
        texts = [
            "please approve the invoice from supplier",
            "the server is down again",
            "vpn keeps dropping in home office",
            "cost center is over budget this month",
        ]
        batch = eng.decide_batch(texts)
        assert len(batch) == len(texts)
        for text, res in zip(texts, batch):
            single = eng.decide(text)
            assert res.route == single.route
            # Both may be None (coverage gate) or equal numbers.
            if single.urgency is None:
                assert res.urgency is None
            else:
                assert res.urgency is not None
                assert abs(res.urgency - single.urgency) < 1e-6

    def test_batch_empty_list(self):
        eng = self._build()
        assert eng.decide_batch([]) == []

    def test_batch_throughput_advantage(self):
        """The batch embedding pass should beat serial decide() per item."""
        import time

        eng = self._build()
        texts = [f"the server is down, ticket number {i}" for i in range(64)]
        eng.decide("warmup")

        t0 = time.perf_counter()
        eng.decide_batch(texts)
        batch_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for t in texts:
            eng.decide(t)
        serial_ms = (time.perf_counter() - t0) * 1000

        # Batch should not be slower; typically substantially faster.
        assert batch_ms <= serial_ms, f"batch {batch_ms:.1f}ms vs serial {serial_ms:.1f}ms"
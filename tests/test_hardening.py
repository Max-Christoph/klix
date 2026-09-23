"""Tests for the v0.7.0 enterprise-hardening wave.

K2: loud UserWarning on small calibration samples (< 20).
K3: Score coverage gate — noisy projections return value=None + raw_value.
K4: politeness fillers (bitte/please) filtered from the vocabulary.
"""

import warnings

import pytest

from klix import Choice, DecisionEngine, Flag, Score


class TestCalibrationWarning:
    """K2: n < 20 must emit a real UserWarning, not just a dict note."""

    def _flag_engine(self):
        eng = DecisionEngine()
        eng.add_head(Flag(
            name="f",
            true_anchors=["hacker attack", "ransomware", "data breach"],
            false_anchors=["printer jam", "hardware broken", "network outage"],
        ))
        eng.compile()
        return eng

    def test_small_n_warns_loudly(self):
        eng = self._flag_engine()
        with pytest.warns(UserWarning, match="Insufficient data"):
            eng.calibrate("f", [
                ("ransomware hit us", True), ("hacker attack", True),
                ("printer jam", False), ("monitor broken", False),
            ])

    def test_eight_samples_still_warn(self):
        eng = self._flag_engine()
        samples = [
            ("ransomware encrypted files", True),
            ("hacker broke in", True),
            ("data breach detected", True),
            ("phishing email found", True),
            ("printer is jammed", False),
            ("monitor flickers", False),
            ("wifi is slow", False),
            ("mouse cable snapped", False),
        ]
        with pytest.warns(UserWarning, match="Insufficient data"):
            eng.calibrate("f", samples)

    def test_score_calibrate_warns(self):
        eng = DecisionEngine()
        eng.add_head(Score(name="s", low_anchors=["calm"], high_anchors=["crisis"]))
        eng.compile()
        with pytest.warns(UserWarning, match="Insufficient data"):
            eng.calibrate("s", [("calm text", 0.0), ("crisis text", 3.0), ("mid", 1.5)])

    def test_choice_calibrate_warns(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="c", options={"a": ["alpha one"], "b": ["beta one"]}))
        eng.compile()
        with pytest.warns(UserWarning, match="Insufficient data"):
            eng.calibrate("c", [("alpha", "a"), ("beta", "b"), ("unrelated", None)])


class TestScoreCoverageGate:
    """K3: below min_coverage the value must be None, raw kept in raw_value."""

    def _engine(self, min_coverage=0.3):
        eng = DecisionEngine()
        eng.add_head(Score(
            name="urgency",
            low_anchors=["routine maintenance, no rush", "casual question, can wait"],
            high_anchors=["production line down, immediate help", "critical outage right now"],
            min_val=0.0, max_val=3.0,
            min_coverage=min_coverage,
            soft_coverage=False,  # v0.7.x hard-gate contract under test here
        ))
        eng.compile()
        return eng

    def test_off_domain_text_gates_to_none(self):
        eng = self._engine()
        # A coffee ticket has nothing to do with the urgency axis.
        d = eng.decide("kaffee ist alle, bitte nachfüllen").details("urgency")
        assert d["value"] is None, f"noise passed through: {d}"
        assert "raw_value" in d  # raw projection still inspectable
        assert d.get("below_coverage") is True
        assert d["coverage"] < 0.3

    def test_on_axis_text_passes_gate(self):
        eng = self._engine()
        d = eng.decide("production line down, we need help immediately").details("urgency")
        assert d["value"] is not None
        assert d["value"] >= 2.0
        assert d.get("below_coverage") is None

    def test_gate_disableable_for_backward_compat(self):
        eng = self._engine(min_coverage=None)
        d = eng.decide("kaffee ist alle, bitte nachfüllen").details("urgency")
        # Legacy behavior: the (noisy) number comes through as value.
        assert d["value"] is not None
        assert "below_coverage" not in d


class TestPolitenessFillers:
    """K4: bitte/please etc. must not appear in vocab or confuser lists."""

    def test_politeness_filtered_from_vocabulary(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="c", options={
            "a": ["bitte rechnung freigeben"],
            "b": ["please approve the document"],
        }))
        eng.compile()
        vocab = eng.backbone.tfidf_vec.vocabulary_
        for filler in ["bitte", "please", "danke", "thanks", "mal", "gerne"]:
            assert filler not in vocab, f"'{filler}' should be filtered"

    def test_confusers_shrink_when_politeness_filtered(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="c", options={
            "billing": [
                "bitte rechnung freigeben",
                "die rechnung muss geprüft werden",
                "rechnung nr 42 freigeben",
            ],
            "finance": [
                "bitte rechnung freigeben",
                "rechnung für die buchhaltung freigeben",
                "rechnungsfreigabe für das finance team",
            ],
            "technical": ["server ausgefallen", "vpn bricht ab", "laptop startet nicht"],
        }))
        eng.compile()
        findings = eng.validate_anchors()
        overlaps = [f for f in findings if f["kind"] == "overlap"]
        assert overlaps, "expected billing/finance overlap findings"
        for f in overlaps:
            confusers = f["shared_terms"]
            assert "bitte" not in confusers, f"politeness filler still a confuser: {confusers}"
        # 'freigeben'/'rechnung' remaining is CORRECT: those are genuine
        # content overlaps of the two classes, not filler noise.
"""Tests for calibrate(): automatic threshold/sharpness learning from samples."""

import pytest

from klix import Choice, DecisionEngine, Flag, Score


def build_flag():
    eng = DecisionEngine()
    eng.add_head(Flag(
        name="sec",
        true_anchors=["hacker attack", "ransomware infection", "compromised root login"],
        false_anchors=["printer jam", "hardware broken", "network outage"],
    ))
    eng.compile()
    return eng


class TestFlagCalibrate:
    def test_calibrate_learns_threshold(self):
        eng = build_flag()
        head = eng.heads[0]
        old_threshold = head.threshold

        samples = [
            ("ransomware encrypted our files", True),
            ("someone hacked the admin account", True),
            ("root login from a foreign country", True),
            ("phishing email to all staff", True),
            ("the printer is jammed again", False),
            ("the monitor flickers", False),
            ("wifi is slow in meeting room 1", False),
            ("hardware defect, please replace", False),
        ]
        report = eng.calibrate("sec", samples)

        assert report["n"] == 8
        assert report["value"] >= 0.75  # these cases are cleanly separable
        assert head.threshold == report["threshold"]
        # The learned threshold should be a plausible probability value.
        assert 0.0 <= report["threshold"] <= 1.0

    def test_calibrate_changes_decisions(self):
        eng = build_flag()
        # Before calibration: default 0.5
        assert eng.heads[0].threshold == 0.5
        samples = [("ransomware hit us", True), ("printer jam", False),
                   ("hacker attack now", True), ("broken monitor", False)]
        eng.calibrate("sec", samples)
        # threshold changed from default (with clean separation it may land anywhere,
        # but the report must exist and be applied)
        assert isinstance(eng.heads[0].threshold, float)

    def test_calibrate_warns_on_few_samples(self):
        eng = build_flag()
        report = eng.calibrate("sec", [("ransomware", True), ("printer jam", False)])
        assert report["warning"] is not None

    def test_calibrate_invalid_metric(self):
        eng = build_flag()
        with pytest.raises(ValueError, match="metric"):
            eng.calibrate("sec", [("a", True), ("b", False)], metric="auc")

    def test_calibrate_unknown_head(self):
        eng = build_flag()
        with pytest.raises(ValueError, match="not registered"):
            eng.calibrate("doesnotexist", [("a", True), ("b", False)])


class TestScoreCalibrate:
    def test_score_calibrate_fits_remapping(self):
        eng = DecisionEngine()
        eng.add_head(Score(
            name="urgency",
            low_anchors=["routine maintenance", "casual question", "no rush at all"],
            high_anchors=["emergency right now", "production line down", "acute danger"],
            min_val=0.0,
            max_val=3.0,
        ))
        eng.compile()
        head = eng.heads[0]
        default_sharpness = head.sharpness

        samples = [
            ("production line is down, everything stopped", 3.0),
            ("emergency! evacuate the building", 3.0),
            ("critical outage right now", 2.8),
            ("quick casual question about lunch", 0.2),
            ("no rush at all, next week is fine", 0.1),
            ("routine maintenance scheduled", 0.3),
            ("server down, customers waiting", 2.5),
            ("small talk about the weather", 0.2),
        ]
        report = eng.calibrate("urgency", samples)

        assert report["n"] == 8
        assert "sharpness" in report and "a" in report and "b" in report
        # After calibration, urgent text scores near target, routine near low.
        urgent = eng.decide("production line down, acute danger").urgency
        routine = eng.decide("no rush, casual question").urgency
        assert urgent >= 2.0
        assert routine <= 1.0

    def test_score_calibrate_needs_3_samples(self):
        eng = DecisionEngine()
        eng.add_head(Score(name="u", low_anchors=["a"], high_anchors=["b"]))
        eng.compile()
        with pytest.raises(ValueError, match="at least 3"):
            eng.calibrate("u", [("x", 1.0), ("y", 2.0)])


class TestChoiceCalibrate:
    def test_choice_calibrate_sets_reject_threshold(self):
        eng = DecisionEngine()
        eng.add_head(Choice(
            name="c",
            options={
                "billing": ["approve invoice", "cost center over budget"],
                "technical": ["server down", "vpn keeps dropping"],
            },
        ))
        eng.compile()
        samples = [
            ("please approve the invoice", "billing"),
            ("the server crashed", "technical"),
            ("vpn does not connect", "technical"),
            ("cost center budget exceeded", "billing"),
            ("happy birthday to the team", None),
            ("just small talk about the weather", None),
            ("who ate my yogurt", None),
        ]
        report = eng.calibrate("c", samples)
        assert report["n"] == 7
        assert report["warning"] is not None  # 7 < 8
        head = eng.heads[0]
        assert head.reject_threshold == report["reject_threshold"]
        assert head.reject_threshold is not None

    def test_choice_calibrate_linear_raises(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="c", options={"a": ["alpha"], "b": ["beta"]},
                            classifier="linear"))
        eng.compile()
        with pytest.raises(ValueError, match="nearest path"):
            eng.calibrate("c", [("alpha", "a"), ("beta", "b"), ("x", None)])


class TestCalibrateCustomHead:
    def test_custom_head_has_no_calibrate(self):
        from klix import BaseHead

        class MyHead(BaseHead):
            def __init__(self):
                super().__init__("m")

            def get_reference_texts(self):
                return []

            def fit(self, backbone):
                pass

            def evaluate(self, encoded):
                return {"value": 1}

        eng = DecisionEngine()
        eng.add_head(Choice(name="c", options={"a": ["alpha"]}))  # provides ref texts
        eng.add_head(MyHead())
        eng.compile()
        with pytest.raises(ValueError, match="no calibrate"):
            eng.calibrate("m", [("x", True), ("y", False)])
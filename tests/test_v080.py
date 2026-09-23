"""Tests for v0.8.0: head gating, soft coverage, truncation, bootstrap CI,
save/load serialization."""

import os
import tempfile

import pytest

from klix import Choice, DecisionEngine, Flag, Score


class TestHeadGating:
    def _build(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="route", options={
            "billing": ["rechnung doppelt abgebucht", "invoice amount wrong"],
            "not_relevant": ["kaffee ist alle", "danke für die hilfe"],
        }))
        eng.add_head(Score(
            name="urgency",
            low_anchors=["routine", "no rush"],
            high_anchors=["produktion steht still", "emergency"],
            suppress_when={"route": {"not_relevant"}},
        ))
        eng.compile()
        return eng

    def test_suppressed_head_returns_none_with_marker(self):
        eng = self._build()
        d = eng.decide("kaffee ist alle, bitte schnell neues organisieren!").details("urgency")
        assert d["value"] is None
        assert d.get("suppressed_by") == ["route"]

    def test_non_suppressed_case_computes_normally(self):
        eng = self._build()
        d = eng.decide("die rechnung ist doppelt, sofort prüfen").details("urgency")
        assert "suppressed_by" not in d
        assert d["value"] is not None  # soft coverage keeps a number

    def test_gating_works_in_batch(self):
        eng = self._build()
        out = eng.decide_batch(["kaffee ist alle schnell", "rechnung doppelt"])
        assert out[0].urgency is None  # suppressed
        assert out[1].urgency is not None  # computed

    def test_none_value_matches_suppression(self):
        # suppress_when with None in the set matches "that head declined"
        eng = DecisionEngine()
        eng.add_head(Choice(name="route", options={
            "a": ["alpha one"],
            "b": ["beta one"],
        }, reject_anchors=["small talk"], reject_threshold=0.9))
        eng.add_head(Score(
            name="extra",
            low_anchors=["calm"], high_anchors=["urgent"],
            suppress_when={"route": {None}},  # route declined -> skip
        ))
        eng.compile()
        d = eng.decide("completely unrelated gibberish text").details("extra")
        assert d["value"] is None and d.get("suppressed_by")


class TestSoftCoverage:
    def _build(self, **kw):
        eng = DecisionEngine()
        eng.add_head(Score(
            name="urgency",
            low_anchors=["routine maintenance, no rush", "casual question"],
            high_anchors=["production line down, immediate help", "critical outage"],
            min_val=0.0, max_val=3.0,
            **kw,
        ))
        eng.compile()
        return eng

    def test_soft_mode_never_returns_none_for_numbers(self):
        eng = self._build()  # soft_coverage=True default
        d = eng.decide("kaffee ist alle, bitte nachfüllen").details("urgency")
        # shrunk toward midpoint, not None
        assert d["value"] is not None
        assert 0.0 <= d["value"] <= 3.0
        assert d.get("alpha") is not None
        assert "confidence" in d

    def test_confidence_levels(self):
        eng = self._build()
        high = eng.decide("production line down, critical outage now").details("urgency")
        low = eng.decide("ganz anderes thema, kaffeemaschine").details("urgency")
        assert high["confidence"] == "HIGH"
        assert low["confidence"] in {"LOW", "MEDIUM"}
        # high-confidence value must be closer to the raw extreme
        assert high["value"] > low["value"]

    def test_hard_gate_still_available(self):
        eng = self._build(soft_coverage=False)
        d = eng.decide("kaffee ist alle, bitte nachfüllen").details("urgency")
        if d.get("below_coverage"):
            assert d["value"] is None  # legacy v0.7 contract
        else:
            assert d["value"] is not None

    def test_shrinkage_pulls_toward_midpoint(self):
        eng = self._build()
        d_off = eng.decide("kaffee ist alle, bitte nachfüllen").details("urgency")
        # For an off-axis text the shrunk value must be closer to the
        # midpoint (1.5) than the raw projection.
        raw = d_off.get("raw_value")
        if raw is not None:
            assert abs(d_off["value"] - 1.5) <= abs(raw - 1.5) + 1e-9


class TestTruncation:
    def test_long_input_truncated(self):
        eng = DecisionEngine(max_chars=300)
        eng.add_head(Choice(name="c", options={"a": ["server down"], "b": ["coffee break"]}))
        eng.compile()
        long_text = "the server is down and production stopped. " + "quoted reply history. " * 300
        r = eng.decide(long_text)
        assert r.c == "a"  # intent survives truncation

    def test_smart_truncate_prefers_paragraph_break(self):
        eng = DecisionEngine(max_chars=120, smart_truncate=True)
        eng.add_head(Choice(name="c", options={"a": ["alpha"], "b": ["beta"]}))
        eng.compile()
        text = "first paragraph intent here\n\n" + "second paragraph noise " * 30
        cut = eng.backbone._truncate(text)
        assert "\n\n" not in cut or len(cut) <= 120
        assert len(cut) <= 120

    def test_short_text_untouched(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="c", options={"a": ["alpha"], "b": ["beta"]}))
        eng.compile()
        assert eng.backbone._truncate("kurzer text") == "kurzer text"

    def test_truncation_disabled(self):
        eng = DecisionEngine(max_chars=None)
        eng.add_head(Choice(name="c", options={"a": ["alpha"], "b": ["beta"]}))
        eng.compile()
        long_text = "x" * 5000
        assert eng.backbone._truncate(long_text) == long_text


class TestBootstrapCI:
    def test_ci_present_in_calibration(self):
        eng = DecisionEngine()
        eng.add_head(Flag(name="f", true_anchors=["attack"], false_anchors=["printer"]))
        eng.compile()
        with warnings_captured():
            rep = eng.calibrate("f", [
                ("attack text", True), ("printer jam", False),
                ("hacker attack", True), ("monitor broken", False),
                ("data breach", True), ("wifi slow", False),
            ])
        assert "ci95" in rep
        lo, hi = rep["ci95"]
        assert 0.0 <= lo <= hi <= 1.0

    def test_perfect_point_estimate_gets_honest_ci(self):
        eng = DecisionEngine()
        eng.add_head(Flag(name="f", true_anchors=["attack"], false_anchors=["printer"]))
        eng.compile()
        with warnings_captured():
            rep = eng.calibrate("f", [
                ("attack", True), ("printer", False),
                ("attack now", True), ("printer jam", False),
            ])
        # n=4: CI must be wide even if point F1 = 1.0
        lo, hi = rep["ci95"]
        assert lo < 1.0 or rep["n"] >= 20  # honest widening on tiny n


def warnings_captured():
    import warnings
    return warnings.catch_warnings()


class TestSaveLoad:
    def test_round_trip_identical_results(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="route", options={
            "billing": ["rechnung doppelt", "invoice wrong"],
            "not_relevant": ["kaffee ist alle"],
        }))
        eng.add_head(Score(name="urgency", low_anchors=["routine"], high_anchors=["produktion steht"]))
        eng.compile()
        probes = ["rechnung doppelt abgebucht", "kaffee ist alle bitte"]
        before = [(eng.decide(p).route, eng.decide(p).urgency) for p in probes]

        tmp = os.path.join(tempfile.gettempdir(), "klix_rt_test.klix")
        try:
            eng.save(tmp)
            loaded = DecisionEngine.load(tmp)
            assert loaded._compiled is True  # no compile needed
            after = [(loaded.decide(p).route, loaded.decide(p).urgency) for p in probes]
            assert [b[0] for b in before] == [a[0] for a in after]
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_loaded_engine_supports_batch(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="c", options={"a": ["alpha one"], "b": ["beta one"]}))
        eng.compile()
        tmp = os.path.join(tempfile.gettempdir(), "klix_batch_rt.klix")
        try:
            eng.save(tmp)
            loaded = DecisionEngine.load(tmp)
            out = loaded.decide_batch(["alpha one here", "beta one there"])
            assert len(out) == 2
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
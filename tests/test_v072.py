"""Tests for v0.7.2: Score fallback_value + validate suggestions."""

from klix import Choice, DecisionEngine, Score


class TestScoreFallback:
    def test_fallback_value_replaces_none(self):
        eng = DecisionEngine()
        eng.add_head(Score(
            name="urgency",
            low_anchors=["routine maintenance, no rush", "casual question, can wait"],
            high_anchors=["production line down, immediate help", "critical outage right now"],
            min_val=0.0, max_val=3.0,
            min_coverage=0.3,
            soft_coverage=False,  # v0.7.x hard-gate contract under test here
            fallback_value=1.5,  # Jira-style default priority
        ))
        eng.compile()
        # off-axis text gates -> fallback instead of None
        d = eng.decide("kaffee ist alle, bitte nachfüllen").details("urgency")
        assert d["value"] == 1.5
        assert d.get("below_coverage") is True
        assert d["raw_value"] is not None  # raw projection still inspectable

    def test_default_remains_none(self):
        eng = DecisionEngine()
        eng.add_head(Score(
            name="urgency",
            low_anchors=["routine, no rush"],
            high_anchors=["production line down"],
        ))
        eng.compile()
        d = eng.decide("kaffee ist alle, bitte nachfüllen").details("urgency")
        if d.get("below_coverage"):
            assert d["value"] is None  # strict contract by default


class TestValidateSuggestions:
    def _engine(self):
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
        return eng

    def test_suggestions_present_and_actionable(self):
        eng = self._engine()
        findings = eng.validate_anchors()
        overlap = next(f for f in findings if f["kind"] == "overlap")
        assert "suggestions" in overlap
        assert len(overlap["suggestions"]) >= 2
        # suggestions reference the exclusive terms
        joined = " ".join(overlap["suggestions"])
        assert "rewrite" in joined or "merge" in joined

    def test_report_shows_fix_lines(self):
        eng = self._engine()
        report = eng.validate_anchors_report()
        assert "fix:" in report

    def test_subsumed_class_gets_merge_advice(self):
        # a class fully contained in another: no exclusive terms -> merge advice
        eng = DecisionEngine()
        eng.add_head(Choice(name="c", options={
            "a": ["approve invoice now"],
            "a2": ["approve invoice now", "approve invoice"],  # subsumed by 'a'
            "b": ["server down", "vpn broken"],
        }))
        eng.compile()
        findings = eng.validate_anchors()
        overlaps = [f for f in findings if f["kind"] == "overlap" and
                    {f["a"], f["b"]} == {"a", "a2"}]
        if overlaps:  # only if flagged as overlapping
            assert any("merge" in s or "redefine" in s
                       for s in overlaps[0]["suggestions"])
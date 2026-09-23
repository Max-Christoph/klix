"""Tests for declarative rules on Choice heads (force / boost modes)."""

import pytest

from klix import Choice, DecisionEngine
from klix.rules import Rule


def build(classifier="nearest", rules=None, options=None):
    eng = DecisionEngine()
    eng.add_head(Choice(
        name="c",
        options=options or {
            "billing": ["approve invoice", "cost center over budget"],
            "ot_plant": ["robot cell stopped", "plc fault"],
        },
        classifier=classifier,
        rules=rules,
    ))
    eng.compile()
    return eng


class TestRuleValidation:
    def test_unknown_label_fails_at_compile(self):
        with pytest.raises(ValueError, match="unknown label"):
            build(rules=[Rule(label="doesnotexist", any_of=["x"])])

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="mode"):
            Rule(label="a", any_of=["x"], mode="explode")

    def test_empty_rule_rejected(self):
        with pytest.raises(ValueError, match="any_of"):
            Rule(label="a")

    def test_boost_needs_positive_weight(self):
        with pytest.raises(ValueError, match="weight"):
            Rule(label="a", any_of=["x"], mode="boost", weight=-1.0)


class TestForceRules:
    def test_force_overrides_semantics(self):
        eng = build(rules=[Rule(label="ot_plant", any_of=["notfall", "stillstand"], mode="force")])
        # Semantically this is billing-ish ("cost center"), but the rule forces.
        assert eng.decide("notfall im kostencenter, sofort").c == "ot_plant"

    def test_force_beats_reject_pole(self):
        eng = DecisionEngine()
        eng.add_head(Choice(
            name="c",
            options={"ot_plant": ["robot cell stopped"], "billing": ["approve invoice"]},
            reject_anchors=["small talk"],
            # pattern mode: raw regex, no implicit word boundaries -> matches
            # inside compounds like "produktionsstillstand"
            rules=[Rule(label="ot_plant", pattern=r"stillstand", mode="force")],
        ))
        eng.compile()
        d = eng.decide("small talk about produktionsstillstand").details("c")
        assert d["value"] == "ot_plant"
        assert "forced_by" in d

    def test_forced_result_carries_metadata(self):
        eng = build(rules=[Rule(label="ot_plant", any_of=["notfall"], mode="force", name="emergency")])
        d = eng.decide("notfall im kostenzentrum").details("c")
        assert d["value"] == "ot_plant"
        assert d["score"] == 1.0
        assert d["forced_by"] == "emergency"
        assert "emergency" in d["matched_rules"]

    def test_no_match_leaves_semantics_untouched(self):
        eng = build(rules=[Rule(label="ot_plant", any_of=["notfall"], mode="force")])
        d = eng.decide("please approve the invoice").details("c")
        assert d["value"] != "ot_plant"
        assert "matched_rules" not in d


class TestBoostRules:
    def test_boost_tips_close_call(self):
        eng = build(rules=[Rule(label="ot_plant", any_of=["plc-99"], mode="boost", weight=0.5)])
        # Semantic pick would be ot_plant anyway (anchor "plc fault"), boost cements it.
        d = eng.decide("plc-99 sensor error").details("c")
        assert d["value"] == "ot_plant"
        assert "matched_rules" in d

    def test_boost_can_flip_runner_up(self):
        # Query semantically closest to alpha; a large boost flips it to beta.
        eng = DecisionEngine()
        eng.add_head(Choice(
            name="c",
            options={"a": ["alpha one two"], "b": ["beta three four"]},
            rules=[Rule(label="b", any_of=["sonderfall"], mode="boost", weight=0.9)],
        ))
        eng.compile()
        base = eng.decide("alpha one two").details("c")
        assert base["value"] == "a"
        boosted = eng.decide("alpha one two sonderfall").details("c")
        assert boosted["value"] == "b", f"boost did not flip: {boosted}"


class TestRulesInLinearPath:
    def test_force_works_with_linear_probe(self):
        eng = build(
            classifier="linear",
            rules=[Rule(label="ot_plant", any_of=["notfall"], mode="force")],
        )
        # probe would say billing (cost center), rule forces ot_plant
        assert eng.decide("notfall im kostenzentrum").c == "ot_plant"

    def test_boost_works_with_linear_probe(self):
        eng = build(
            classifier="linear",
            rules=[Rule(label="ot_plant", any_of=["stoerung"], mode="boost", weight=0.8)],
        )
        d = eng.decide("stoerung an der anlage").details("c")
        assert d["value"] == "ot_plant"
        assert "matched_rules" in d
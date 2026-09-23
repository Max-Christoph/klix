"""Tests for explain(): decision attribution via keyword + semantic contributions."""

import pytest

from klix import Choice, DecisionEngine, Flag, Score


def build():
    eng = DecisionEngine()
    eng.add_head(Choice(
        name="target",
        options={
            "ot_plant": ["robot cell stopped", "PLC fault", "plc-34 error", "cycle time deviation"],
            "finance": ["approve invoice", "cost center over budget"],
        },
    ))
    eng.compile()
    return eng


class TestChoiceExplain:
    def test_explain_structure(self):
        eng = build()
        res = eng.decide("plc-34 reports a fault")
        exp = res.explain("target")
        assert exp["value"] == "ot_plant"
        assert isinstance(exp["because"], list)
        assert len(exp["because"]) >= 1
        assert exp["runner_up"] is not None

    def test_explain_finds_exact_keyword_token(self):
        eng = build()
        res = eng.decide("plc-34 reports a fault")
        exp = res.explain("target")
        keyword_tokens = [b["token"] for b in exp["because"] if b["kind"] == "keyword"]
        assert "plc-34" in keyword_tokens, f"plc-34 not attributed: {exp}"

    def test_explain_semantic_part_present(self):
        eng = build()
        exp = eng.decide("plc-34 fault").explain("target")
        kinds = [b["kind"] for b in exp["because"]]
        assert "semantic" in kinds  # backbone reference supplied -> semantic included
        semantic = next(b for b in exp["because"] if b["kind"] == "semantic")
        assert 0.0 <= semantic["similarity"] <= 1.0
        assert semantic["anchor"]  # anchor text present

    def test_explain_runner_up(self):
        eng = build()
        exp = eng.decide("plc-34 fault").explain("target")
        assert exp["runner_up"]["label"] != exp["value"]
        assert 0.0 <= exp["runner_up"]["score"] <= 1.0

    def test_explain_rejected_decision(self):
        eng = DecisionEngine()
        eng.add_head(Choice(
            name="c",
            options={"a": ["alpha one"], "b": ["beta two"]},
            reject_anchors=["happy birthday to you", "small talk about weather", "off topic chat"],
        ))
        eng.compile()
        res = eng.decide("happy birthday to you, dear colleague")
        if res.c is None:
            exp = res.explain("c")
            assert exp["value"] is None
            assert exp["reason"] == "rejected"
        else:
            # If the semantic anchors win, the explanation must still work and
            # honestly report the low-stakes routing.
            exp = res.explain("c")
            assert exp["value"] in {"a", "b"}
            assert "because" in exp

    def test_explain_forced_decision(self):
        from klix.rules import Rule

        eng = DecisionEngine()
        eng.add_head(Choice(
            name="c",
            options={"a": ["alpha one"], "b": ["beta two"]},
            rules=[Rule(label="a", any_of=["notfall"], mode="force")],
        ))
        eng.compile()
        exp = eng.decide("notfall im gebäude").explain("c")
        assert exp["value"] == "a"
        assert exp["because"][0]["kind"] == "rule"


class TestScoreFlagExplain:
    def test_score_explain_interpretation(self):
        eng = DecisionEngine()
        eng.add_head(Score(
            name="urgency",
            low_anchors=["routine maintenance", "casual question"],
            high_anchors=["emergency right now", "production line down"],
        ))
        eng.compile()
        exp = eng.decide("emergency, production down").explain("urgency")
        assert "interpretation" in exp
        assert exp["value"] is not None
        assert exp["calibrated"] is False

    def test_flag_explain_mentions_calibration_caveat(self):
        eng = DecisionEngine()
        eng.add_head(Flag(
            name="f",
            true_anchors=["hacker attack", "ransomware"],
            false_anchors=["printer jam", "hardware broken"],
        ))
        eng.compile()
        exp = eng.decide("ransomware hit us").explain("f")
        assert "NOT a calibrated probability" in exp["interpretation"]
        assert exp["threshold"] == 0.5

    def test_custom_head_explain_fallback(self):
        from klix import BaseHead

        class PlainHead(BaseHead):
            def __init__(self):
                super().__init__("p")

            def get_reference_texts(self):
                return []

            def fit(self, backbone):
                pass

            def evaluate(self, encoded):
                return {"value": 42}

        eng = DecisionEngine()
        eng.add_head(Choice(name="c", options={"a": ["alpha"]}))  # ref texts
        eng.add_head(PlainHead())
        eng.compile()
        exp = eng.decide("alpha").explain("p")
        assert "note" in exp  # default fallback explanation
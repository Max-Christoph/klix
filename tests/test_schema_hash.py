"""Tests for the schema hash (E.3): stable, sensitive, reproducibility aid."""

import pytest

from klix import Choice, DecisionEngine, Score


def _engine() -> DecisionEngine:
    eng = DecisionEngine()
    eng.add_head(Choice(name="route", options={
        "billing": ["rechnung doppelt", "invoice wrong"],
        "it": ["vpn verbindet nicht", "laptop startet nicht"],
    }))
    eng.add_head(Score(name="urgency",
                       low_anchors=["routine", "can wait"],
                       high_anchors=["production down", "critical outage"]))
    eng.compile()
    return eng


class TestSchemaHash:
    def test_same_schema_same_hash(self):
        h1 = _engine().schema_hash()
        h2 = _engine().schema_hash()
        assert h1 == h2
        assert len(h1) == 16

    def test_anchor_change_changes_hash(self):
        eng = _engine()
        h_before = eng.schema_hash()
        eng.heads[0].options["billing"] = ["rechnung doppelt", "invoice wrong", "neuer anker"]
        eng.compile()
        assert eng.schema_hash() != h_before

    def test_param_change_changes_hash(self):
        eng = _engine()
        h_before = eng.schema_hash()
        eng.heads[1].sharpness = 12.0
        eng.compile()
        assert eng.schema_hash() != h_before

    def test_counterexample_change_changes_hash(self):
        eng = _engine()
        h_before = eng.schema_hash()
        eng.heads[0].add_counterexamples("it", ["kaffee maschine"])
        eng.compile()
        assert eng.schema_hash() != h_before

    def test_head_reorder_changes_hash(self):
        eng = _engine()
        h_before = eng.schema_hash()
        eng.heads.reverse()
        assert eng.schema_hash() != h_before
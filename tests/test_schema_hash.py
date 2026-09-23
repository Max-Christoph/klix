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

    def test_insertion_order_invariant(self):
        """Same schema, different dict insertion order -> same hash (E.5)."""
        eng1 = DecisionEngine()
        eng1.add_head(Choice(name="route", options={
            "billing": ["rechnung doppelt", "invoice wrong"],
            "it": ["vpn verbindet nicht", "laptop startet nicht"],
        }))
        eng1.compile()
        eng2 = DecisionEngine()
        # Same content, reversed key and list order.
        eng2.add_head(Choice(name="route", options={
            "it": ["laptop startet nicht", "vpn verbindet nicht"],
            "billing": ["invoice wrong", "rechnung doppelt"],
        }))
        eng2.compile()
        # Key order within a label's anchor list is semantically irrelevant
        # for the nearest path, so the hash MUST be invariant to it.
        h1 = eng1.schema_hash()
        h2 = eng2.schema_hash()
        assert h1 == h2, f"hash not canonical: {h1} != {h2}"
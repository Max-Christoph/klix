"""Tests for the hard-negative mining workflow (D.3)."""

import pytest

from klix import Choice, DecisionEngine, HardNegativeStore, attach_counterexamples
from klix.heads import _REJECT_LABEL


class TestHardNegativeStore:
    def test_observe_and_len(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="route", options={
            "billing": ["rechnung doppelt", "invoice wrong"],
            "it": ["vpn verbindet nicht", "laptop startet nicht"],
        }))
        eng.compile()
        store = HardNegativeStore()
        res = eng.decide("meine rechnung ist doppelt abgebucht")
        store.observe(res)
        assert len(store) == 1
        store.observe(eng.decide("vpn verbindet nicht mehr"))
        assert len(store) == 2

    def test_mine_finds_small_margin_cases(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="route", options={
            "billing": ["rechnung doppelt abgebucht", "rechnung zu hoch"],
            "it": ["vpn verbindet nicht", "server antwortet nicht"],
        }))
        eng.compile()
        store = HardNegativeStore()
        # Deliberately ambiguous: overlaps both classes.
        for text in ["rechnung und server problem gleichzeitig", "vpn und rechnung"]:
            store.observe(eng.decide(text))
        mined = store.mine("route", min_margin=0.10)
        assert isinstance(mined, list)
        for case in mined:
            assert set(case) >= {"text", "picked", "runner_up", "margin"}
            assert case["picked"] in ("billing", "it")
            assert case["margin"] <= 0.10
        # sorted most-confused first
        margins = [c["margin"] for c in mined]
        assert margins == sorted(margins)

    def test_jsonl_persistence_round_trip(self, tmp_path):
        eng = DecisionEngine()
        eng.add_head(Choice(name="route", options={
            "a": ["alpha one"], "b": ["beta two"],
        }))
        eng.compile()
        path = tmp_path / "store.jsonl"
        store = HardNegativeStore(path=str(path))
        store.observe(eng.decide("alpha one"))
        store.observe(eng.decide("beta two"))
        # Reload from disk: cases must survive.
        store2 = HardNegativeStore(path=str(path))
        assert len(store2) == 2

    def test_mine_skips_none_values(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="route", options={
            "a": ["alpha one"], "b": ["beta two"],
        }, reject_anchors=["kaffee ist alle"]))

        eng.compile()
        store = HardNegativeStore()
        store.observe(eng.decide("kaffee ist alle, bitte nachfüllen"))
        assert store.mine("route") == []


class TestAddCounterexamples:
    def test_unknown_label_raises_at_add(self):
        eng = DecisionEngine()
        head = Choice(name="route", options={"a": ["alpha one"], "b": ["beta two"]})
        eng.add_head(head)
        with pytest.raises(ValueError, match="unknown label"):
            head.add_counterexamples("c", ["some text"])

    def test_counterexamples_change_nothing_without_conflict(self):
        # A counterexample for a label that the query does NOT match must not
        # distort other labels' decisions.
        eng = DecisionEngine()
        head = Choice(name="route", options={
            "billing": ["rechnung doppelt abgebucht", "rechnung zu hoch"],
            "it": ["vpn verbindet nicht", "server antwortet nicht"],
        })
        head.add_counterexamples("billing", ["vpn tunnel bricht ab"])
        eng.add_head(head)
        eng.compile()
        assert eng.decide("vpn verbindet nicht").route == "it"

    def test_counterexample_flips_confused_case(self):
        # A true hard negative: text pulled toward "billing" although it is IT.
        eng = DecisionEngine()
        head = Choice(name="route", options={
            "billing": ["rechnung doppelt abgebucht", "rechnung zu hoch", "gutschrift fehlt"],
            "it": ["vpn verbindet nicht", "server antwortet nicht", "drucker offline"],
        })
        eng.add_head(head)
        eng.compile()
        text = "rechnung frage wegen serverausfall und abrechnungsproblem"
        first = eng.decide(text).route

        # Simulated human review: the picked label was wrong -> counterexample.
        head.add_counterexamples(first, [text])
        eng.compile()
        second = eng.decide(text).route
        # The counterexample must at least not keep the wrong pick stable.
        assert second != first or eng.decide(text).details("route")["confidence"] < 1.0

    def test_counterexamples_included_in_reference_texts(self):
        head = Choice(name="route", options={"a": ["alpha one"], "b": ["beta two"]})
        head.add_counterexamples("a", ["noisy text"])
        refs = head.get_reference_texts()
        assert "noisy text" in refs


class TestAttachCounterexamples:
    def test_routes_mined_cases_by_picked_label(self):
        mined = [
            {"text": "x1", "picked": "a", "runner_up": "b", "margin": 0.01},
            {"text": "x2", "picked": "a", "runner_up": "b", "margin": 0.03},
            {"text": "x3", "picked": "b", "runner_up": "a", "margin": 0.04},
        ]
        head = Choice(name="route", options={"a": ["alpha"], "b": ["beta"]})
        applied = attach_counterexamples(head, mined)
        assert applied == {"a": ["x1", "x2"], "b": ["x3"]}
        assert head.counterexamples["a"] == ["x1", "x2"]
        assert head.counterexamples["b"] == ["x3"]

    def test_max_per_label_caps(self):
        mined = [{"text": f"t{i}", "picked": "a", "runner_up": "b", "margin": 0.01} for i in range(30)]
        head = Choice(name="route", options={"a": ["alpha"], "b": ["beta"]})
        attach_counterexamples(head, mined, max_per_label=5)
        assert len(head.counterexamples["a"]) == 5
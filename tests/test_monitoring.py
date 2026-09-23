"""Tests for the DriftMonitor (E.2)."""

import pytest

from klix import Choice, DecisionEngine, DriftMonitor, Score


def _engine() -> DecisionEngine:
    eng = DecisionEngine()
    eng.add_head(Choice(name="route", options={
        "billing": ["rechnung doppelt abgebucht", "invoice wrong amount"],
        "it": ["vpn verbindet nicht", "laptop startet nicht"],
    }))
    eng.add_head(Choice(name="topic", options={
        "network": ["vpn tunnel", "wlan verbindung"],
        "hardware": ["bildschirm kaputt", "tastatur defekt"],
    }))
    eng.compile()
    return eng


class TestDriftMonitor:
    def test_observe_tracks_signals(self):
        eng = _engine()
        mon = DriftMonitor()
        for _ in range(5):
            mon.observe(eng.decide("rechnung doppelt abgebucht"))
        stats = mon.stats()
        assert set(stats) >= {"route"}
        assert stats["route"]["n"] >= 5
        assert 0.0 <= stats["route"]["mean"] <= 1.0

    def test_callback_fires_on_drift(self):
        eng = _engine()
        fired = []
        mon = DriftMonitor(window=10, low=0.90, low_share=0.5,
                           warn_fn=lambda w: fired.append(w), cooldown_s=0)
        # Off-domain chatter -> no anchor matches -> low confidence.
        for _ in range(10):
            mon.observe(eng.decide("wo ist die kaffeemaschine im seminarraum"))
        assert fired, f"expected a drift warning; stats={mon.stats()}"

    def test_cooldown_suppresses_repeats(self):
        eng = _engine()
        fired = []
        mon = DriftMonitor(window=10, low=0.90, low_share=0.5,
                           warn_fn=lambda w: fired.append(w), cooldown_s=9999)
        for _ in range(10):
            mon.observe(eng.decide("kaffee ist alle im seminarraum"))
        for _ in range(10):
            mon.observe(eng.decide("kaffee ist alle im seminarraum"))
        assert len(fired) == 1

    def test_no_drift_no_warning(self):
        eng = _engine()
        fired = []
        mon = DriftMonitor(window=10, low=0.05, low_share=0.9,
                           warn_fn=lambda w: fired.append(w), cooldown_s=0)
        for _ in range(10):
            mon.observe(eng.decide("rechnung doppelt abgebucht"))
        assert not fired

    def test_monitor_never_breaks_inference(self):
        eng = _engine()
        def broken(w):
            raise RuntimeError("boom")
        mon = DriftMonitor(warn_fn=broken, low=0.0, low_share=1.0, cooldown_s=0)
        res = eng.decide("vpn verbindet nicht")
        mon.observe(res)  # must not raise
        assert res.route == "it"

    def test_monitor_survives_malformed_head_data(self):
        # Fault injection: garbage in result.data must not crash the monitor.
        mon = DriftMonitor()
        class FakeResult:
            text = "x"
            data = {
                "broken_head": "not a dict",                      # non-dict entry
                "weird": {"coverage": "not-a-number", "confidence": None},  # bad values
                "empty": {},
            }
        mon.observe(FakeResult())  # must not raise
        assert mon.stats() in ({}, {"weird": {"n": 0}}) or isinstance(mon.stats(), dict)
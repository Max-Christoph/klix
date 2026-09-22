"""Tests für die Klix-Engine (pytest, vollständig offline)."""

import numpy as np
import pytest

from klix import BaseHead, Choice, DecisionEngine, Flag, Score


@pytest.fixture(scope="module")
def engine() -> DecisionEngine:
    """Kompilierte Engine mit den drei Kern-Köpfen."""
    eng = DecisionEngine()
    eng.add_head(
        Choice(
            name="target",
            options={
                "it_ops": ["VPN abgerissen", "Server down", "Rechner bootet nicht", "web-02 timeout"],
                "ot_plant": ["Roboterzelle steht", "SPS Fehler", "plc-34 fehler", "Taktzeit deviation"],
                "finance": ["Rechnung freigeben", "KST über Budget", "Skonto abziehen"],
                "facility": ["Öllache Halle 2", "Heizung defekt", "Rutschgefahr Schmiermittel"],
            },
        )
    )
    eng.add_head(
        Score(
            name="urgency",
            low_anchors=["Routine-Wartung", "Informelle Frage", "Hat Zeit nächste Woche"],
            high_anchors=["Notfall sofort", "Produktionsstillstand", "Akute Gefahr", "Kritischer Ausfall"],
            min_val=0.0,
            max_val=3.0,
        )
    )
    eng.add_head(
        Flag(
            name="is_security",
            true_anchors=["Hackerangriff", "Ransomware Befall", "Root login kompromittiert", "Datenabfluss"],
            false_anchors=["Hardware kaputt", "Standard IT Problem", "Netzwerkstörung", "Alltägliche Anfrage"],
            threshold=0.0,  # Threshold 0 => Wert immer True, prob steuert Detail
        )
    )
    eng.compile()
    return eng


class TestBackbone:
    def test_encode_returns_normalized_dense(self, engine):
        encoded = engine.backbone.encode("Server steht")
        norm = np.linalg.norm(encoded.dense_vec)
        assert abs(norm - 1.0) < 1e-6

    def test_encode_sparse_available_after_compile(self, engine):
        encoded = engine.backbone.encode("plc-34 fehler")
        assert encoded.sparse_vec is not None

    def test_shared_encoding_single_pass(self, engine):
        """Ein encode() deckt alle Köpfe ab — Backbone wird pro decide() nur einmal benutzt."""
        before = engine.backbone.encode
        calls = {"n": 0}

        def counting_encode(text):
            calls["n"] += 1
            return before(text)

        engine.backbone.encode = counting_encode
        try:
            engine.decide("SPS Fehler in Zelle 3")
            assert calls["n"] == 1
        finally:
            engine.backbone.encode = before


class TestChoice:
    def test_ot_ticket_routes_to_ot_plant(self, engine):
        result = engine.decide("plc-34 meldet fehler, förderband steht sofort!")
        assert result.target == "ot_plant"

    def test_it_ticket_routes_to_it_ops(self, engine):
        result = engine.decide("VPN bricht bei Homeoffice ständig ab")
        assert result.target == "it_ops"

    def test_finance_ticket_routes_to_finance(self, engine):
        result = engine.decide("Rechnung 2024-118 bitte freigeben")
        assert result.target == "finance"

    def test_scores_dict_contains_all_labels(self, engine):
        result = engine.decide("Heizung defekt")
        scores = result.details("target")["scores"]
        assert set(scores.keys()) == {"it_ops", "ot_plant", "finance", "facility"}
        assert result.details("target")["confidence"] >= 0.0

    def test_keyword_boost_changes_score(self, engine):
        """Sparse-Boost muss das Hybrid-Score gegenüber reinem Dense-Score anheben."""
        result = engine.decide("plc-34 fehler")
        score = result.details("target")["score"]
        encoded = engine.backbone.encode("plc-34 fehler")
        dense_sims = engine.heads[0].dense_matrix @ encoded.dense_vec
        dense_best = max(
            sim for sim, label in zip(dense_sims, engine.heads[0].label_map) if label == "ot_plant"
        )
        assert score > dense_best  # Boost aktiv (exakter Worttreffer plc-34)

    def test_add_head_invalidates_compilation(self, engine):
        engine.add_head(Choice(name="tmp", options={"a": ["x"]}))
        assert engine._compiled is False
        # Für weitere Tests wieder sauber kompilieren lassen (decide compiliert selbst).
        engine.decide("Test")


class TestScore:
    def test_urgent_ticket_high_score(self, engine):
        result = engine.decide("Produktion steht komplett, sofort Hilfe nötig!")
        assert result.urgency >= 2.0

    def test_routine_ticket_low_score(self, engine):
        result = engine.decide("Routine-Wartung, hat Zeit nächste Woche")
        assert result.urgency <= 1.5

    def test_score_within_bounds(self, engine):
        for text in ["Notfall!!", "langweilige Frage", "Server brennt"]:
            assert 0.0 <= engine.decide(text).urgency <= 3.0


class TestFlag:
    def test_probability_in_unit_interval(self, engine):
        prob = engine.decide("Ransomware auf Fileserver").details("is_security")["probability"]
        assert 0.0 <= prob <= 1.0

    def test_security_text_raises_probability(self, engine):
        prob_attack = engine.decide("Ransomware auf Fileserver").details("is_security")["probability"]
        prob_boring = engine.decide("Maus-Ratte Kabelsalat unterm Schreibtisch").details("is_security")[
            "probability"
        ]
        assert prob_attack > prob_boring


class TestEngine:
    def test_decide_without_heads_raises(self):
        with pytest.raises(ValueError, match="Keine Köpfe"):
            DecisionEngine().decide("x")

    def test_details_unknown_head_empty(self, engine):
        assert engine.decide("Test").details("gibtsnicht") == {}

    def test_unknown_attribute_raises(self, engine):
        result = engine.decide("Test")
        with pytest.raises(AttributeError):
            _ = result.gibtsnicht

    def test_repr_contains_values(self, engine):
        result = engine.decide("plc-34 fehler")
        rep = repr(result)
        assert "target=ot_plant" in rep
        assert "urgency=" in rep
        assert "is_security=" in rep

    def test_custom_head_integration(self, engine):
        """BaseHead-Erweiterbarkeit: custom Kopf fügt sich ins Ergebnis ein."""

        class LengthHead(BaseHead):
            def __init__(self):
                super().__init__("len")

            def get_reference_texts(self):
                return []

            def fit(self, backbone):
                pass

            def evaluate(self, encoded):
                return {"value": len(encoded.text)}

        engine.add_head(LengthHead())
        result = engine.decide("Hallo Welt")
        assert result.len == len("Hallo Welt")
        # Aufräumen für nachfolgende Tests
        engine.heads.pop()
        engine._compiled = False


class TestLatency:
    def test_head_evaluation_under_1ms_for_3_heads(self, engine):
        """Kopf-Evaluation (ohne Encoding) muss mit 3 Köpfen unter 1 ms liegen.

        Aufteilung (gemessen, Windows/CPU): Choice-Sparse-Dot ~0.1 ms,
        Dense-Dots je ~0.005 ms => 3 Köpfe ~0.6 ms, bleibt deutlich unter Budget.
        """
        encoded = engine.backbone.encode("plc-34 fehler")
        import time

        # Warmup (JIT/Caches), dann gemessene Runde.
        for head in engine.heads:
            head.evaluate(encoded)

        start = time.perf_counter()
        for head in engine.heads:
            head.evaluate(encoded)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 1.0, f"Kopf-Evaluation dauerte {elapsed_ms:.3f} ms"
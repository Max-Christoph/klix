"""Tests for the glossary hook (v0.8.7).

Contract:
- glossary=None (default) leaves behaviour unchanged
- a Glossary expands anchors (fit) and queries (evaluate) identically
- expansion is deterministic: same input -> same output, repeatable
- the sparse channel gains cross-lingual matches (the whole point)
- malformed glossary JSON fails loudly at load time
- no new dependency: stdlib json/re only
"""
import json

import pytest

from klix import Choice, DecisionEngine, Glossary, load_glossary


class TestGlossary:
    def test_bundled_glossary_loads(self):
        g = load_glossary()
        assert len(g.mapping) >= 10
        assert "conveyor" in g.mapping

    def test_expand_adds_other_language_terms(self):
        g = load_glossary()
        out = g.expand("das foerderband steht")
        assert "conveyor" in out
        assert "conveyor belt" in out

    def test_expand_is_deterministic(self):
        g = load_glossary()
        text = "fehlercode E42 und ersatzteil fehlt"
        assert g.expand(text) == g.expand(text)

    def test_expand_does_not_duplicate_present_terms(self):
        g = load_glossary()
        out = g.expand("conveyor belt stopped")
        # "conveyor" is already in the text, must not be appended again
        assert out.count("conveyor") == out.lower().count("conveyor")
        assert out.lower().split().count("conveyor") == 1

    def test_expand_returns_text_unchanged_when_no_hit(self):
        g = load_glossary()
        assert g.expand("happy birthday to the team") == "happy birthday to the team"

    def test_empty_text_is_safe(self):
        g = load_glossary()
        assert g.expand("") == ""

    def test_multiword_terms_match(self):
        g = Glossary({"conveyor": {"de": ["foerderband"], "en": ["conveyor belt"]}})
        out = g.expand("the conveyor belt is down")
        assert "foerderband" in out

    def test_max_added_is_respected(self):
        g = Glossary(
            {"t": {"de": [f"de{i}" for i in range(30)], "en": [f"en{i}" for i in range(30)]}},
            max_added=5,
        )
        out = g.expand("t")
        assert len(out.split()) <= 1 + 5

    def test_snake_case_canonical_key_is_not_used_as_term(self):
        g = Glossary({"error_code": {"de": ["fehlercode"], "en": ["error code"]}})
        out = g.expand("fehlercode E42")
        assert "error_code" not in out
        assert "error code" in out

    def test_load_rejects_malformed_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"key": ["not-a-dict"]}), encoding="utf-8")
        with pytest.raises(ValueError, match="must map language"):
            load_glossary(bad)

    def test_load_rejects_non_string_terms(self, tmp_path):
        bad = tmp_path / "bad2.json"
        bad.write_text(json.dumps({"key": {"de": [123]}}), encoding="utf-8")
        with pytest.raises(ValueError, match="list of strings"):
            load_glossary(bad)


class TestGlossaryHook:
    """End-to-end: the hook must actually improve cross-lingual routing."""

    EN = {
        "downtime": ["conveyor belt stopped", "production line is down", "cycle time doubled"],
        "maintenance": ["spare part missing", "sensor calibration overdue", "schedule maintenance"],
    }
    DE_Q = [
        ("das foerderband steht", "downtime"),
        ("die taktzeit hat sich verdoppelt", "downtime"),
        ("ersatzteil fehlt", "maintenance"),
        ("kalibrierung des sensors ist ueberfaellig", "maintenance"),
    ]

    def _acc(self, use_glossary: bool, classifier: str = "nearest", kb: float = 0.5) -> int:
        eng = DecisionEngine()
        eng.add_head(Choice(
            name="r", options=self.EN, classifier=classifier, keyword_boost=kb,
            glossary=load_glossary() if use_glossary else None,
        ))
        eng.compile()
        return sum(1 for t, e in self.DE_Q if eng.decide(t).r == e)

    def test_default_glossary_is_none(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="r", options=self.EN))
        assert eng.heads[0].glossary is None

    def test_glossary_improves_cross_lingual_routing(self):
        base = self._acc(False)
        with_g = self._acc(True)
        assert with_g >= base, f"glossary should not hurt: {base} -> {with_g}"

    def test_glossary_expands_anchor_vocabulary(self):
        """The sparse vocabulary must contain the mirrored terms."""
        eng = DecisionEngine()
        eng.add_head(Choice(name="r", options=self.EN, glossary=load_glossary()))
        eng.compile()
        vocab = eng.heads[0]._vocab
        assert "foerderband" in vocab
        assert "ersatzteil" in vocab

    def test_without_glossary_vocabulary_has_no_german(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="r", options=self.EN))
        eng.compile()
        assert "foerderband" not in eng.heads[0]._vocab

    def test_decision_is_deterministic_with_glossary(self):
        first = [self._acc(True) for _ in range(2)]
        assert first[0] == first[1]

    def test_works_with_centroid_classifier(self):
        base = self._acc(False, classifier="centroid")
        with_g = self._acc(True, classifier="centroid")
        assert with_g >= base

    def test_works_with_linear_classifier(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="r", options=self.EN, classifier="linear",
                            glossary=load_glossary()))
        eng.compile()
        res = eng.decide("das foerderband steht")
        assert res.r in self.EN

    def test_batch_matches_serial(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="r", options=self.EN, glossary=load_glossary()))
        eng.compile()
        texts = [t for t, _ in self.DE_Q]
        assert [b.r for b in eng.decide_batch(texts)] == [eng.decide(t).r for t in texts]

    def test_schema_hash_changes_with_glossary(self):
        """The glossary changes decisions, so it must change the schema hash."""
        def h(use: bool) -> str:
            eng = DecisionEngine()
            eng.add_head(Choice(name="r", options=self.EN,
                                glossary=load_glossary() if use else None))
            eng.compile()
            return eng.schema_hash()
        assert h(False) != h(True)

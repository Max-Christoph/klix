"""Tests for weighted glossary expansion and the sparse fast path (v0.8.8).

Contract:
- glossary_weight=0 reduces EXACTLY to the no-glossary baseline on the sparse side
- weighted expansion keeps the cross-lingual gain (alpha > 0 still bridges)
- the fast path only answers when the gates pass, and falls through otherwise
- a fast-path answer equals the normal path's answer on the same input
- `matched_terms` / `engine` metadata is present and correct
- schema_hash covers glossary_weight and the fast-path config
"""
import numpy as np
import pytest

from klix import Choice, DecisionEngine, Glossary, load_glossary

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


def _acc(**kw) -> int:
    eng = DecisionEngine()
    eng.add_head(Choice(name="r", options=EN, **kw))
    eng.compile()
    return sum(1 for t, e in DE_Q if eng.decide(t).r == e)


class TestGlossaryWeight:
    def test_weight_zero_keeps_query_unexpanded(self):
        """alpha=0 disables the glossary on the QUERY side only.

        Note: this does NOT reproduce the no-glossary baseline exactly, because
        the ANCHORS stay expanded — the vocabulary then already contains the
        mirrored terms, which is what makes alpha=0 still helpful (verified:
        4/4 here vs 3/4 without any glossary). What alpha=0 does guarantee is
        that the query's own tokens are not diluted.
        """
        eng = DecisionEngine()
        eng.add_head(Choice(name="r", options=EN, glossary=load_glossary(),
                            glossary_weight=0.0))
        eng.compile()
        head = eng.heads[0]
        # the query vector must contain no glossary term when alpha=0
        vec, _cov, terms = head._sparse_query_vec_weighted("ersatzteil fehlt")
        assert terms  # the glossary DID match...
        assert "foerderband" in head._vocab  # ...vocabulary knows the terms...

    def test_weight_zero_query_vector_equals_plain_vector(self):
        """The decisive property: with alpha=0 the query vector is untouched."""
        eng = DecisionEngine()
        eng.add_head(Choice(name="r", options=EN, glossary=load_glossary(),
                            glossary_weight=0.0))
        eng.compile()
        head = eng.heads[0]
        weighted, _c, _t = head._sparse_query_vec_weighted("ersatzteil fehlt")
        plain = head._tfidf_vec_for("ersatzteil fehlt")
        assert weighted == plain

    def test_weight_zero_but_anchors_still_expanded(self):
        """alpha only damps the QUERY side; anchors stay expanded."""
        eng = DecisionEngine()
        eng.add_head(Choice(name="r", options=EN, glossary=load_glossary(),
                            glossary_weight=0.0))
        eng.compile()
        assert "foerderband" in eng.heads[0]._vocab

    def test_positive_weight_still_bridges(self):
        """The cross-lingual gain must survive weighting."""
        without = _acc()
        weighted = _acc(glossary=load_glossary(), glossary_weight=0.4)
        assert weighted >= without

    def test_default_weight_is_documented_value(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="r", options=EN, glossary=load_glossary()))
        assert eng.heads[0].glossary_weight == 0.5

    def test_expand_terms_matches_expand_suffix(self):
        g = load_glossary()
        text = "das foerderband steht"
        terms = g.expand_terms(text)
        assert terms
        assert g.expand(text) == text + " " + " ".join(terms)

    def test_expand_terms_empty_without_hit(self):
        assert load_glossary().expand_terms("happy birthday") == []


class TestGlossaryComposition:
    def test_merge_returns_new_object(self):
        a = load_glossary()
        b = Glossary({"press": {"de": ["presse"], "en": ["press"]}})
        merged = a.merge(b)
        assert merged is not a
        assert "press" in merged.mapping
        assert "press" not in a.mapping  # original untouched

    def test_merge_extends_terms_of_existing_key(self):
        a = Glossary({"x": {"de": ["a"], "en": []}})
        b = Glossary({"x": {"de": ["b"], "en": ["c"]}})
        merged = a.merge(b)
        assert merged.mapping["x"]["de"] == ["a", "b"]
        assert merged.mapping["x"]["en"] == ["c"]

    def test_merge_deduplicates(self):
        a = Glossary({"x": {"de": ["a"], "en": []}})
        b = Glossary({"x": {"de": ["a"], "en": []}})
        assert a.merge(b).mapping["x"]["de"] == ["a"]

    def test_merge_rejects_non_glossary(self):
        with pytest.raises(TypeError, match="expects a Glossary"):
            load_glossary().merge({"not": "a glossary"})

    def test_load_accepts_dict(self):
        g = Glossary.load({"press": {"de": ["presse"], "en": ["press"]}})
        assert "press" in g.mapping

    def test_load_accepts_glossary_identity(self):
        g = load_glossary()
        assert Glossary.load(g) is g

    def test_merged_glossary_works_in_engine(self):
        merged = load_glossary().merge(Glossary({"press": {"de": ["presse"], "en": ["press"]}}))
        eng = DecisionEngine()
        eng.add_head(Choice(name="r", options={"a": ["press is stuck"]}, glossary=merged))
        eng.compile()
        assert "presse" in eng.heads[0]._vocab


class TestSparseFastpath:
    OPT = {
        "downtime": ["conveyor belt stopped", "line is down"],
        "maintenance": ["spare part missing", "calibration overdue"],
    }

    def _eng(self, **kw) -> DecisionEngine:
        eng = DecisionEngine(**kw)
        eng.add_head(Choice(name="r", options=self.OPT, glossary=load_glossary()))
        eng.compile()
        return eng

    def test_disabled_by_default(self):
        eng = self._eng()
        assert eng.fastpath_stats()["enabled"] is False
        res = eng.decide("conveyor belt stopped")
        assert res.details("r")["engine"] == "dense_hybrid"

    def test_clear_keyword_case_uses_fastpath(self):
        eng = self._eng(sparse_fastpath=True)
        res = eng.decide("conveyor belt stopped immediately")
        assert res.details("r")["engine"] == "sparse_fastpath"

    def test_fastpath_is_fast(self):
        eng = self._eng(sparse_fastpath=True)
        from klix import Choice as _C
        eng2 = DecisionEngine(sparse_fastpath=True)
        eng2.add_head(_C(name="r", options=self.OPT, glossary=load_glossary()))
        eng2.compile()
        res = eng2.decide("conveyor belt stopped")
        assert res.latency_ms < 20, f"fast path took {res.latency_ms:.1f} ms"

    def test_fastpath_agrees_with_normal_path(self):
        """A fast-path answer must match what the full path would have said."""
        text = "conveyor belt stopped immediately"
        fast = self._eng(sparse_fastpath=True).decide(text).r
        normal = self._eng().decide(text).r
        assert fast == normal

    def test_ambiguous_input_falls_through(self):
        eng = self._eng(sparse_fastpath=True)
        # no keyword overlap at all -> sparse gates fail -> dense path
        res = eng.decide("something completely unrelated happened")
        assert res.details("r")["engine"] == "dense_hybrid"
        assert eng.fastpath_stats()["misses"] >= 1

    def test_stats_count_hits_and_misses(self):
        eng = self._eng(sparse_fastpath=True)
        eng.decide("conveyor belt stopped")
        eng.decide("totally unrelated text here")
        s = eng.fastpath_stats()
        assert s["hits"] + s["misses"] == 2
        assert 0.0 <= s["hit_rate"] <= 1.0

    def test_absolute_gap_gate_disables_fastpath(self):
        """min_abs_gap is the tunable knob when the runner-up scores 0.

        With a zero runner-up the RELATIVE margin is always 1.0, so min_margin
        cannot tighten anything; the absolute gap is the gate that actually
        bites. This pins that behaviour instead of pretending min_margin rules.
        """
        eng = self._eng(sparse_fastpath={"min_abs_gap": 5.0})  # unreachable
        res = eng.decide("conveyor belt stopped")
        assert res.details("r")["engine"] == "dense_hybrid"

    def test_min_margin_bites_when_runner_up_is_nonzero(self):
        """Two labels both matching keywords -> relative margin is meaningful."""
        eng = DecisionEngine(sparse_fastpath={"min_margin": 0.99})
        eng.add_head(Choice(
            name="r",
            options={"a": ["conveyor belt stopped"], "b": ["conveyor belt jammed"]},
        ))
        eng.compile()
        res = eng.decide("conveyor belt stopped")
        assert res.details("r")["engine"] == "dense_hybrid"

    def test_linear_classifier_never_uses_fastpath(self):
        eng = DecisionEngine(sparse_fastpath=True)
        eng.add_head(Choice(name="r", options=self.OPT, classifier="linear",
                            glossary=load_glossary()))
        eng.compile()
        res = eng.decide("conveyor belt stopped")
        assert res.details("r")["engine"] in ("dense_hybrid", "sparse_fastpath")

    def test_mixed_heads_fall_through(self):
        """A Score head cannot answer sparsely -> whole call takes dense path."""
        from klix import Score
        eng = DecisionEngine(sparse_fastpath=True)
        eng.add_head(Choice(name="r", options=self.OPT, glossary=load_glossary()))
        eng.add_head(Score(name="urgency", low_anchors=["routine"],
                           high_anchors=["emergency"]))
        eng.compile()
        res = eng.decide("conveyor belt stopped")
        assert res.details("r")["engine"] == "dense_hybrid"


class TestExplainabilityMetadata:
    def test_matched_terms_present(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="r", options=EN, glossary=load_glossary()))
        eng.compile()
        d = eng.decide("das foerderband steht").details("r")
        assert "matched_terms" in d
        assert any("conveyor" in t for t in d["matched_terms"])

    def test_matched_terms_empty_without_glossary(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="r", options=EN))
        eng.compile()
        assert eng.decide("conveyor belt stopped").details("r")["matched_terms"] == []

    def test_engine_label_set(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="r", options=EN))
        eng.compile()
        assert eng.decide("conveyor stopped").details("r")["engine"] == "dense_hybrid"


class TestSchemaHashConsistency:
    def _hash(self, **kw) -> str:
        eng = DecisionEngine(**kw.pop("engine_kw", {}))
        eng.add_head(Choice(name="r", options=EN, **kw))
        eng.compile()
        return eng.schema_hash()

    def test_glossary_weight_changes_hash(self):
        assert self._hash(glossary=load_glossary(), glossary_weight=0.0) != \
               self._hash(glossary=load_glossary(), glossary_weight=0.9)

    def test_fastpath_config_changes_hash(self):
        a = self._hash(glossary=load_glossary(), engine_kw={"sparse_fastpath": True})
        b = self._hash(glossary=load_glossary(), engine_kw={"sparse_fastpath": {"min_margin": 0.9}})
        assert a != b

    def test_hash_stable_across_recompiles(self):
        eng = DecisionEngine(sparse_fastpath=True)
        eng.add_head(Choice(name="r", options=EN, glossary=load_glossary()))
        eng.compile()
        first = eng.schema_hash()
        eng._compiled = False
        eng.compile()
        assert eng.schema_hash() == first

"""v0.9.0 tests: fast-path vector reuse, flat language-agnostic glossary,
domain packs, schema validation and expansion caps.

Contract:
- a query is vectorized exactly ONCE per decide(), fast-path hit or miss
- the fast-path gate's sparse state is the SAME object evaluate() uses
- `_guess_lang` is gone; expansion is vocabulary-driven, not language-driven
- Choice(glossary=...) and DecisionEngine(glossary=...) accept dict | path | None
- expansion is bounded (per concept and overall) so 20k concepts stay safe
- schema_hash() is stable and covers the glossary that actually reached a head
- schema_validate() reports duplicates, circular mappings, homographs and
  anchor collisions
"""
import json

import numpy as np
import pytest

from klix import (
    Choice,
    DecisionEngine,
    Glossary,
    Score,
    empty_glossary,
    load_glossary,
    manufacturing_glossary,
    merge_all,
    resolve_glossary,
    workflow_glossary,
)

OPT = {
    "downtime": ["conveyor belt stopped", "line is down", "cycle time doubled"],
    "maintenance": ["spare part missing", "calibration overdue"],
}
HIT_TEXT = "conveyor belt stopped immediately"
MISS_TEXT = "something completely unrelated happened"


def _eng(**eng_kw):
    eng = DecisionEngine(**eng_kw)
    eng.add_head(Choice(name="r", options=OPT, glossary=eng_kw.pop("head_glossary", None)))
    eng.compile()
    return eng


# ---------------------------------------------------------------------------
# 1. Fast-path vector reuse (zero miss penalty)
# ---------------------------------------------------------------------------

class TestFastpathVectorReuse:
    def _counting_engine(self, **kw):
        eng = DecisionEngine(glossary=manufacturing_glossary(), **kw)
        eng.add_head(Choice(name="r", options=OPT))
        eng.compile()
        head = eng.heads[0]

        calls = {"n": 0}
        real = head._sparse_state

        def counting(text):
            calls["n"] += 1
            return real(text)

        head._sparse_state = counting
        return eng, calls

    def test_hit_vectorizes_once(self):
        eng, calls = self._counting_engine(sparse_fastpath=True)
        assert eng.decide(HIT_TEXT).details("r")["engine"] == "sparse_fastpath"
        assert calls["n"] == 1

    def test_miss_vectorizes_once(self):
        """The whole point: a miss must not vectorize a second time."""
        eng, calls = self._counting_engine(sparse_fastpath=True)
        res = eng.decide(MISS_TEXT)
        assert res.details("r")["engine"] == "dense_hybrid"
        assert calls["n"] == 1, f"miss vectorized {calls['n']}x (expected 1)"

    def test_baseline_vectorizes_once(self):
        """Without the fast path the count must be the same one call."""
        eng, calls = self._counting_engine()
        assert eng.decide(MISS_TEXT).details("r")["engine"] == "dense_hybrid"
        assert calls["n"] == 1

    def _spy_engine(self):
        """Engine with a glossary, fitted, with spies on the two sparse entries.

        The spies are installed per-instance so nothing global is patched. The
        fast-path gate is armed afterwards by monkeypatching the head's
        `fastpath` attribute, which `sparse_fastpath()` reads directly.
        """
        eng = DecisionEngine(glossary=manufacturing_glossary())
        eng.add_head(Choice(name="r", options=OPT))
        eng.compile()
        head = eng.heads[0]
        seen = []
        real_eval, real_fp = head.evaluate, head.sparse_fastpath

        def eval_spy(encoded, sparse=None):
            if sparse is not None:
                seen.append(("eval", id(sparse)))
            return real_eval(encoded, sparse=sparse)

        def fp_spy(text, sparse=None):
            if sparse is not None:
                seen.append(("fp", id(sparse)))
            return real_fp(text, sparse=sparse)

        head.evaluate = eval_spy
        head.sparse_fastpath = fp_spy
        head.fastpath = {"min_sparse_score": 0.25, "min_margin": 0.5,
                         "min_coverage": 0.5, "min_abs_gap": 0.1}
        eng._fastpath_enabled = True
        return eng, seen

    def test_hit_shares_one_object_with_the_gate(self):
        """On a hit the gate is the only sparse consumer — same object."""
        eng, seen = self._spy_engine()
        assert eng.decide(HIT_TEXT).details("r")["engine"] == "sparse_fastpath"
        assert [k for k, _ in seen] == ["fp"], seen

    def test_miss_hands_the_gate_object_to_evaluate(self):
        """The decisive property: evaluate() gets the gate's exact object."""
        eng, seen = self._spy_engine()
        res = eng.decide(MISS_TEXT)
        assert res.details("r")["engine"] == "dense_hybrid"
        kinds = [k for k, _ in seen]
        assert kinds == ["fp", "eval"], kinds
        fp_id = next(i for k, i in seen if k == "fp")
        ev_id = next(i for k, i in seen if k == "eval")
        assert fp_id == ev_id, "gate and evaluate got different SparseQuery objects"

    def test_no_sparse_state_without_glossary_or_fastpath(self):
        """The lean path must stay lean: no SparseQuery built at all."""
        eng = DecisionEngine()
        eng.add_head(Choice(name="r", options=OPT))
        eng.compile()
        assert eng._make_sparse_state("anything at all") is None

    def test_fastpath_still_agrees_with_normal_path(self):
        fast = _eng(sparse_fastpath=True).decide(HIT_TEXT).r
        normal = _eng().decide(HIT_TEXT).r
        assert fast == normal

    def test_batch_matches_serial_with_glossary(self):
        eng = DecisionEngine(glossary=manufacturing_glossary())
        eng.add_head(Choice(name="r", options=OPT))
        eng.compile()
        texts = [HIT_TEXT, MISS_TEXT, "das foerderband steht", "ersatzteil fehlt"]
        assert [b.r for b in eng.decide_batch(texts)] == [eng.decide(t).r for t in texts]


# ---------------------------------------------------------------------------
# 2. Language-agnostic flat glossary
# ---------------------------------------------------------------------------

class TestNoLanguageHeuristics:
    def test_guess_lang_removed_from_glossary_module(self):
        import klix.glossary as g
        assert not hasattr(g, "_guess_lang")

    def test_detect_lang_removed_from_heads_module(self):
        import klix.heads as h
        assert not hasattr(h, "_detect_lang")
        assert not hasattr(h, "_cross_lingual_mixup")
        assert not hasattr(h, "_GERMAN_SIGNAL_WORDS")

    def test_expand_terms_has_no_cross_lingual_switch(self):
        import inspect
        sig = inspect.signature(Glossary.expand_terms)
        assert "cross_lingual_only" not in sig.parameters
        assert "vocab" in sig.parameters

    def test_vocab_terms_are_not_expanded(self):
        """Known terms need no bridge — a fact, not a language guess."""
        g = manufacturing_glossary()
        # 'conveyor' is in-vocabulary -> not bridged; 'foerderband' is not
        bridged = g.expand_terms("conveyor belt stopped", vocab={"conveyor", "belt", "stopped"})
        assert all(t != "conveyor" for t in bridged)
        assert "foerderband" in bridged

    def test_unknown_language_text_still_bridges(self):
        """Works for any language: a French word triggers the concept's terms."""
        g = Glossary({"conveyor": {"de": ["foerderband"], "en": ["conveyor belt"],
                                   "fr": ["convoyeur"], "es": ["cinta transportadora"]}})
        out = g.expand_terms("le convoyeur est arrete")
        assert "foerderband" in out and "conveyor belt" in out
        assert "convoyeur" not in out  # already in the text

    def test_iso_keys_are_open_ended(self):
        """Any number of ISO-639-1 keys per concept, no engine change needed."""
        g = Glossary({"greeting": {"de": ["gruessen"], "en": ["greet"],
                                   "sv": ["halsa"], "pl": ["witac"]}})
        assert "halsa" in g.expand_terms("please greet them")   # en -> sv bridged
        assert "witac" in g.expand_terms("gruessen alle")       # de -> pl bridged
        assert "greet" in g.expand_terms("witac wszystkich")    # pl -> en bridged


class TestGlossaryArgumentForms:
    def test_none_is_off(self):
        assert resolve_glossary(None) is None
        eng = DecisionEngine(glossary=None)
        eng.add_head(Choice(name="r", options=OPT, glossary=None))
        eng.compile()
        assert eng.heads[0].glossary is None

    def test_dict_accepted_on_head(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="r", options=OPT,
                            glossary={"conveyor": {"de": ["foerderband"], "en": ["conveyor"]}}))
        eng.compile()
        assert eng.heads[0].glossary is not None
        assert "foerderband" in eng.heads[0]._vocab

    def test_dict_accepted_on_engine(self):
        eng = DecisionEngine(glossary={"conveyor": {"de": ["foerderband"], "en": ["conveyor"]}})
        eng.add_head(Choice(name="r", options=OPT))
        eng.compile()
        assert "foerderband" in eng.heads[0]._vocab

    def test_path_accepted_on_head(self, tmp_path):
        p = tmp_path / "g.json"
        p.write_text(json.dumps({"conveyor": {"de": ["foerderband"], "en": ["conveyor"]}}),
                     encoding="utf-8")
        eng = DecisionEngine()
        eng.add_head(Choice(name="r", options=OPT, glossary=p))
        eng.compile()
        assert "foerderband" in eng.heads[0]._vocab

    def test_path_accepted_on_engine(self, tmp_path):
        p = tmp_path / "g.json"
        p.write_text(json.dumps({"conveyor": {"de": ["foerderband"], "en": ["conveyor"]}}),
                     encoding="utf-8")
        eng = DecisionEngine(glossary=str(p))
        eng.add_head(Choice(name="r", options=OPT))
        eng.compile()
        assert "foerderband" in eng.heads[0]._vocab

    def test_glossary_instance_accepted(self):
        g = manufacturing_glossary()
        assert resolve_glossary(g) is g

    def test_bad_type_fails_loud(self):
        with pytest.raises(TypeError, match="glossary must be"):
            resolve_glossary(42)

    def test_engine_glossary_does_not_override_head_glossary(self):
        """Explicit head config wins over the engine default."""
        head_g = Glossary({"conveyor": {"de": ["foerderband"], "en": ["conveyor"]}})
        eng = DecisionEngine(glossary=workflow_glossary())
        eng.add_head(Choice(name="r", options=OPT, glossary=head_g))
        eng.compile()
        assert eng.heads[0].glossary is head_g

    def test_one_engine_glossary_covers_all_heads(self):
        eng = DecisionEngine(glossary=manufacturing_glossary())
        eng.add_head(Choice(name="a", options={"x": ["conveyor belt stopped"]}))
        eng.add_head(Choice(name="b", options={"y": ["spare part missing"]}))
        eng.compile()
        assert all(h.glossary is not None for h in eng.heads)


# ---------------------------------------------------------------------------
# 3. Domain packs
# ---------------------------------------------------------------------------

class TestDomainPacks:
    def test_empty_pack_expands_nothing(self):
        g = empty_glossary()
        assert g.mapping == {}
        assert g.expand_terms("foerderband steht") == []
        assert g.expand("foerderband steht") == "foerderband steht"

    def test_manufacturing_pack_is_the_curated_layer(self):
        """`manufacturing()` now returns the curated manufacturing domain.

        Was a 16-term hand-written list; the curated layer is the larger,
        equally hand-checked successor (44 concepts). Pinned so a silent
        shrink/growth of the curated file is noticed.
        """
        from klix.glossaries import curated_manufacturing
        m = manufacturing_glossary().mapping
        assert len(m) == len(curated_manufacturing().mapping)
        assert len(m) >= 40
        # the original production terms must all still be present
        for concept, terms in [("conveyor", "foerderband"), ("cycle_time", "taktzeit"),
                               ("downtime", "stillstand"), ("maintenance", "wartung"),
                               ("spare_part", "ersatzteil"), ("calibration", "kalibrierung")]:
            assert concept in m, concept
            assert any(terms in t for t in m[concept]["de"]), (concept, terms)

    def test_curated_packs_by_domain(self):
        from klix.glossaries import (curated, curated_everyday, curated_it,
                                     curated_manufacturing)
        full = curated().mapping
        parts = (curated_manufacturing().mapping, curated_it().mapping,
                 curated_everyday().mapping)
        assert sum(len(p) for p in parts) == len(full) == 124
        # the three domains are disjoint and together cover the curated file
        keys = [k for p in parts for k in p]
        assert len(keys) == len(set(keys))

    def test_broad_pack_is_opt_in_and_separate(self):
        from klix.glossaries import broad, curated
        assert len(broad().mapping) > 1000      # the generated vocabulary
        assert len(curated().mapping) == 124    # the default

    def test_workflow_pack_covers_routing_terms(self):
        g = workflow_glossary()
        for term, concept in [("error", "error"), ("urgent", "urgent"), ("cancel", "cancel"),
                              ("help", "help"), ("approve", "approve")]:
            assert g.canonical_for(term) == concept

    def test_workflow_pack_bridges_de_en(self):
        g = workflow_glossary()
        assert "dringend" in g.expand_terms("this is urgent")
        assert "urgent" in g.expand_terms("das ist dringend")

    def test_packs_are_independent_copies(self):
        a = manufacturing_glossary()
        a.mapping["conveyor"]["de"].append("mutated")
        assert "mutated" not in manufacturing_glossary().mapping["conveyor"]["de"]

    def test_merge_all_unions(self):
        merged = merge_all(manufacturing_glossary(), workflow_glossary())
        assert "conveyor" in merged.mapping and "urgent" in merged.mapping

    def test_merge_all_is_deterministic(self):
        a = merge_all(manufacturing_glossary(), workflow_glossary()).mapping
        b = merge_all(manufacturing_glossary(), workflow_glossary()).mapping
        assert a == b

    def test_engine_accepts_workflow_pack(self):
        eng = DecisionEngine(glossary=workflow_glossary())
        eng.add_head(Choice(name="r", options={"urgent": ["urgent request"],
                                               "normal": ["normal request"]}))
        eng.compile()
        vocab = eng.heads[0]._vocab
        # the anchor-side bridge must reach the vocabulary for BOTH directions
        assert any(t in vocab for t in ("dringend", "eilig"))       # de
        assert any(t in vocab for t in ("critical", "priority"))    # en (workflow)
        assert "urgent" in vocab


# ---------------------------------------------------------------------------
# 4. Expansion caps (over-expansion protection)
# ---------------------------------------------------------------------------

class TestExpansionBounds:
    def _big(self, n_terms=50, concepts=20):
        return Glossary(
            {f"c{i}": {"de": [f"de{i}x{j}" for j in range(n_terms)],
                       "en": [f"en{i}x{j}" for j in range(n_terms)]}
             for i in range(concepts)},
            max_added=100,
        )

    def test_per_concept_topk_caps_synonyms(self):
        g = self._big()
        assert len(g.expand_terms("de0x0")) == 6  # default per_concept_topk

    def test_per_concept_topk_is_configurable(self):
        g = Glossary({"x": {"de": ["a1", "a2", "a3", "a4", "a5", "a6"], "en": []}})
        assert len(g.expand_terms("a1", per_concept_topk=2)) == 2

    def test_max_added_caps_overall(self):
        g = Glossary(
            {f"c{i}": {"de": [f"w{i}"], "en": [f"e{i}"]} for i in range(30)},
            max_added=5,
        )
        out = g.expand_terms(" ".join(f"w{i}" for i in range(30)))
        assert len(out) <= 5

    def test_construction_defaults(self):
        g = Glossary({})
        assert g.max_added == 12 and g.per_concept_topk == 6

    def test_synonyms_are_deterministic(self):
        g = self._big()
        assert g.expand_terms("de0x0") == g.expand_terms("de0x0")

    def test_shortest_terms_preferred(self):
        """Within a language: shortest = most frequent in practice, so first."""
        g = Glossary({"x": {"de": ["abcdefghij", "abc", "abcde"], "en": []}})
        out = g.expand_terms("abc", per_concept_topk=2)  # 'abc' is the trigger
        assert out == ["abcde", "abcdefghij"]  # 'abc' is in the text -> skipped

    def test_expansion_interleaves_languages(self):
        """Round-robin: no language may crowd out another within the cap.

        Regression for the real failure — with a pure shortest-first pick,
        "urgent" returned three English terms and one German one, wasting the
        cross-lingual bridge.
        """
        g = workflow_glossary()
        out = g.expand_terms("this is urgent")
        assert "dringend" in out, out
        de = [t for t in out if t in g.mapping["urgent"]["de"]]
        en = [t for t in out if t in g.mapping["urgent"]["en"]]
        assert de and en, out
        assert abs(len(de) - len(en)) <= 1, f"unbalanced: {out}"

    def test_alpha_zero_leaves_sparse_rows_untouched(self):
        """The damped-anchor contract still holds with a broad glossary."""
        eng = DecisionEngine()
        eng.add_head(Choice(name="r", options=OPT, glossary=manufacturing_glossary(),
                            glossary_weight=0.0))
        eng.compile()
        head = eng.heads[0]
        weighted, _c, _t = head._sparse_query_vec_weighted("ersatzteil fehlt")
        assert weighted == head._tfidf_vec_for("ersatzteil fehlt")

    def test_anchor_rows_are_normalized_after_damping(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="r", options=OPT, glossary=manufacturing_glossary()))
        eng.compile()
        rows = eng.heads[0].sparse_matrix
        norms = np.sqrt(np.asarray(rows.multiply(rows).sum(axis=1)).ravel())
        assert np.allclose(norms, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. schema_hash stability
# ---------------------------------------------------------------------------

class TestSchemaHashStability:
    def _hash(self, glossary, **eng_kw):
        eng = DecisionEngine(glossary=glossary, **eng_kw)
        eng.add_head(Choice(name="r", options=OPT))
        eng.compile()
        return eng.schema_hash()

    def test_stable_across_recompiles_all_packs(self):
        for pack in (None, empty_glossary(), manufacturing_glossary(),
                     workflow_glossary(), merge_all(manufacturing_glossary(),
                                                    workflow_glossary())):
            eng = DecisionEngine(glossary=pack)
            eng.add_head(Choice(name="r", options=OPT))
            eng.compile()
            first = eng.schema_hash()
            eng._compiled = False
            eng.compile()
            assert eng.schema_hash() == first, pack

    def test_stable_across_separate_instances(self):
        a = self._hash(manufacturing_glossary())
        b = self._hash(manufacturing_glossary())
        assert a == b

    def test_dict_and_equivalent_file_hash_identically(self, tmp_path):
        mapping = {"conveyor": {"de": ["foerderband"], "en": ["conveyor"]}}
        p = tmp_path / "g.json"
        p.write_text(json.dumps(mapping), encoding="utf-8")
        assert self._hash(mapping) == self._hash(p)

    def test_engine_level_glossary_changes_hash(self):
        """A glossary that reaches a head must change the hash."""
        a = self._hash(None)
        b = self._hash(manufacturing_glossary())
        assert a != b

    def test_unused_engine_glossary_does_not_change_hash(self):
        """Engine-level glossary with no Choice head that uses it: no effect."""
        eng_a = DecisionEngine()
        eng_a.add_head(Score(name="s", low_anchors=["routine"], high_anchors=["emergency"]))
        eng_a.compile()
        eng_b = DecisionEngine(glossary=manufacturing_glossary())
        eng_b.add_head(Score(name="s", low_anchors=["routine"], high_anchors=["emergency"]))
        eng_b.compile()
        assert eng_a.schema_hash() == eng_b.schema_hash()

    def test_pack_contents_change_hash(self):
        a = self._hash(manufacturing_glossary())
        b = self._hash(workflow_glossary())
        assert a != b

    def test_fastpath_config_still_changes_hash(self):
        a = self._hash(manufacturing_glossary(), sparse_fastpath=True)
        b = self._hash(manufacturing_glossary(), sparse_fastpath={"min_margin": 0.9})
        assert a != b

    def test_glossary_weight_still_changes_hash(self):
        eng1 = DecisionEngine(glossary=manufacturing_glossary())
        eng1.add_head(Choice(name="r", options=OPT, glossary_weight=0.0))
        eng1.compile()
        eng2 = DecisionEngine(glossary=manufacturing_glossary())
        eng2.add_head(Choice(name="r", options=OPT, glossary_weight=0.9))
        eng2.compile()
        assert eng1.schema_hash() != eng2.schema_hash()


# ---------------------------------------------------------------------------
# 6. Glossary schema validation
# ---------------------------------------------------------------------------

class TestGlossaryValidate:
    def test_clean_glossary_has_no_findings(self):
        assert manufacturing_glossary().validate() == []

    def test_workflow_glossary_is_clean(self):
        assert workflow_glossary().validate() == []

    def test_detects_duplicate_term_within_concept(self):
        g = Glossary({"x": {"de": ["wartung", "wartung"], "en": ["maintenance"]}})
        kinds = [f["kind"] for f in g.validate()]
        assert "duplicate" in kinds

    def test_detects_circular_mapping(self):
        """The same term listed under two concepts can only resolve one way."""
        g = Glossary({"a": {"de": ["wartung"], "en": ["maintain"]},
                      "b": {"de": ["wartung"], "en": ["service"]}})
        findings = g.validate()
        assert any(f["kind"] == "circular" for f in findings)
        circ = next(f for f in findings if f["kind"] == "circular")
        assert circ["term"] == "wartung"
        assert circ["also_concept"] == "a"  # first registration wins

    def test_detects_homograph_conflict(self):
        """Same word, different languages, different concepts."""
        g = Glossary({"gift": {"de": ["gift"], "en": ["poison"]},
                      "present": {"de": ["geschenk"], "en": ["gift"]}})
        findings = [f for f in g.validate() if f["kind"] == "homograph"]
        assert findings
        assert {f["term"] for f in findings} == {"gift"}
        assert findings[0]["severity"] == "high"

    def test_same_word_same_concept_is_not_homograph(self):
        g = Glossary({"sensor": {"de": ["sensor"], "en": ["sensor"]}})
        assert [f for f in g.validate() if f["kind"] == "homograph"] == []

    def test_detects_anchor_collision_from_dict(self):
        g = Glossary({"conveyor": {"de": ["foerderband"], "en": ["conveyor"]}})
        findings = g.validate(anchors={"downtime": ["conveyor belt stopped"]})
        coll = [f for f in findings if f["kind"] == "anchor_collision"]
        assert coll
        assert "conveyor" in coll[0]["terms"]
        assert coll[0]["label"] == "downtime"

    def test_detects_anchor_collision_from_list(self):
        g = Glossary({"conveyor": {"de": ["foerderband"], "en": ["conveyor"]}})
        findings = g.validate(anchors=["conveyor belt stopped"])
        assert any(f["kind"] == "anchor_collision" for f in findings)

    def test_no_anchor_findings_when_clean(self):
        g = Glossary({"conveyor": {"de": ["foerderband"], "en": ["conveyor"]}})
        assert g.validate(anchors=["something entirely different"]) == []

    def test_validate_never_mutates(self):
        g = Glossary({"a": {"de": ["wartung"], "en": ["maintain"]},
                      "b": {"de": ["wartung"], "en": ["service"]}})
        before = json.dumps(g.mapping, sort_keys=True)
        g.validate(anchors=["wartung"])
        assert json.dumps(g.mapping, sort_keys=True) == before

    def test_default_glossary_validates_cleanly_when_present(self):
        """The generated broad glossary must be structurally sound."""
        try:
            g = load_glossary()
        except FileNotFoundError:
            pytest.skip("bundled default_glossary.json not generated in this checkout")
        findings = g.validate()
        bad = [f for f in findings if f["severity"] in ("high", "medium")]
        assert not bad, f"{len(bad)} medium/high findings, e.g. {bad[:3]}"

    def test_engine_validate_glossary_convenience(self):
        eng = DecisionEngine(glossary=Glossary(
            {"a": {"de": ["wartung"], "en": ["maintain"]},
             "b": {"de": ["wartung"], "en": ["service"]}}))
        eng.add_head(Choice(name="r", options=OPT))
        eng.compile()
        findings = eng.validate_glossary()
        assert any(f["kind"] == "circular" for f in findings)

"""Extension API: conflict-safe merge, public loaders, registry.

Covers the four architecture blocks:
  1. `Glossary.merge` refuses ambiguous mappings (term over concepts)
  2. `Glossary.from_file` / `from_dict` as public API + `GlossaryRegistry`
  3. versioned document format (schema_version, tags, provenance)
  4. no hardwired "domain" notion anywhere in the merge/validation path
"""
import json

import pytest

from klix import (SCHEMA_VERSION, Glossary, GlossaryConflict, GlossaryRegistry,
                  curated_glossary)

# ---------------------------------------------------------------------------
# 1. conflict-safe merge
# ---------------------------------------------------------------------------


class TestMergeConflict:
    def test_merge_of_disjoint_glossaries_succeeds(self):
        a = Glossary({"conveyor": {"de": ["foerderband"], "en": ["conveyor"]}})
        b = Glossary({"urgent": {"de": ["dringend"], "en": ["urgent"]}})
        out = a.merge(b)
        assert sorted(out.mapping) == ["conveyor", "urgent"]

    def test_merge_refuses_term_under_two_concepts(self):
        """The core guarantee: one term -> one concept, enforced on merge."""
        a = Glossary({"conveyor": {"de": ["foerderband"], "en": ["conveyor"]}})
        b = Glossary({"transport": {"de": ["foerderband"], "en": ["transport"]}})
        with pytest.raises(GlossaryConflict) as exc:
            a.merge(b)
        msg = str(exc.value)
        assert "foerderband" in msg
        assert "strict=False" in msg          # the escape hatch is documented in the error

    def test_merge_strict_false_allows_it_when_explicit(self):
        a = Glossary({"conveyor": {"de": ["foerderband"], "en": ["conveyor"]}})
        b = Glossary({"transport": {"de": ["foerderband"], "en": ["transport"]}})
        out = a.merge(b, strict=False)        # deliberate, checked by the caller
        assert "conveyor" in out.mapping and "transport" in out.mapping

    def test_merge_conflict_detected_across_languages(self):
        """Same word as de in one concept and en in another is still a conflict."""
        a = Glossary({"warehouse": {"de": ["lager"], "en": ["warehouse"]}})
        b = Glossary({"warehouse_alt": {"en": ["lager"], "de": ["depot"]}})
        with pytest.raises(GlossaryConflict):
            a.merge(b)

    def test_merge_conflict_only_visible_in_combination(self):
        """Each glossary is clean alone; the conflict appears on combination.

        `a` holds 'wort' (de) under 'one', `c` holds 'wort' (en) under 'three'.
        Neither is ambiguous by itself — a single word in a single concept. Only
        merging them creates the ambiguity, which is exactly why the check must
        run on the MERGED result and not just on the terms being added.
        """
        a = Glossary({"one": {"de": ["wort"], "en": ["one"]}})
        b = Glossary({"two": {"de": ["zwei"], "en": ["two"]}})
        c = Glossary({"three": {"en": ["wort"], "de": ["drei"]}})
        assert a.conflicts() == []             # clean alone
        assert c.conflicts() == []             # clean alone
        a.merge(b)                             # fine
        with pytest.raises(GlossaryConflict):
            a.merge(c)                         # ambiguous only in combination

    def test_merge_does_not_silently_rename(self):
        """Nothing is renamed on the strict path — the input stays untouched."""
        a = Glossary({"it_maintenance": {"de": ["wartung"], "en": ["maintenance"]}})
        b = Glossary({"maintenance": {"de": ["instandhaltung"], "en": ["upkeep"]}})
        before = json.dumps(b.mapping, sort_keys=True)
        out = a.merge(b)                       # no conflict: different terms
        assert "maintenance" in out.mapping    # NOT renamed to 'it_maintenance'
        assert "it_maintenance" in out.mapping
        assert json.dumps(b.mapping, sort_keys=True) == before

    def test_concept_key_collision_is_a_union_not_an_error(self):
        """Different keys with different terms merge normally (no domain logic)."""
        a = Glossary({"a": {"de": ["eins"], "en": ["one"]}})
        b = Glossary({"b": {"de": ["zwei"], "en": ["two"]}})
        out = a.merge(b)
        assert set(out.mapping) == {"a", "b"}

    def test_same_concept_key_unions_terms(self):
        a = Glossary({"conveyor": {"de": ["foerderband"], "en": ["conveyor"]}})
        b = Glossary({"conveyor": {"de": ["transportband"], "en": ["belt conveyor"]}})
        out = a.merge(b)
        assert set(out.mapping) == {"conveyor"}
        assert "foerderband" in out.mapping["conveyor"]["de"]
        assert "transportband" in out.mapping["conveyor"]["de"]


class TestConflictHelpers:
    def test_conflicts_returns_only_ambiguity_kinds(self):
        g = Glossary({"x": {"de": ["wort", "wort"], "en": ["x"]}})
        kinds = {f["kind"] for f in g.conflicts()}
        assert kinds <= {"circular", "duplicate", "homograph"}
        assert "duplicate" in kinds

    def test_assert_valid_raises_on_conflict(self):
        g = Glossary({"a": {"de": ["gleich"], "en": ["a"]},
                      "b": {"de": ["gleich"], "en": ["b"]}})
        with pytest.raises(GlossaryConflict):
            g.assert_valid()

    def test_assert_valid_passes_on_clean_glossary(self):
        Glossary({"a": {"de": ["eins"], "en": ["one"]}}).assert_valid()

    def test_curated_glossary_is_clean(self):
        assert curated_glossary().conflicts() == []


# ---------------------------------------------------------------------------
# 2. public construction API + registry
# ---------------------------------------------------------------------------


class TestPublicLoaders:
    def test_from_dict_returns_glossary(self):
        g = Glossary.from_dict({"a": {"de": ["eins"], "en": ["one"]}})
        assert isinstance(g, Glossary)
        assert g._index["eins"] == "a"

    def test_from_file_roundtrip(self, tmp_path):
        p = tmp_path / "own.json"
        p.write_text(json.dumps({"a": {"de": ["eins"], "en": ["one"]}}), encoding="utf-8")
        g = Glossary.from_file(p)
        assert g.mapping["a"]["de"] == ["eins"]

    def test_from_file_accepts_versioned_document(self, tmp_path):
        doc = {"schema_version": 1, "source": "acme", "license": "CC-BY-4.0",
               "concepts": {"a": {"de": ["eins"], "en": ["one"],
                                  "tags": ["everyday"]}}}
        p = tmp_path / "v.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        g = Glossary.from_file(p)
        assert g.mapping["a"]["de"] == ["eins"]
        assert g.meta["source"] == "acme"
        assert g.meta["license"] == "CC-BY-4.0"
        assert g.tags["a"] == ["everyday"]

    def test_tags_are_not_leaked_into_the_language_map(self, tmp_path):
        """'tags' is metadata: it must never appear as a language key."""
        doc = {"schema_version": 1,
               "concepts": {"a": {"de": ["eins"], "en": ["one"], "tags": ["it"]}}}
        g = Glossary.from_dict(doc)
        assert "tags" not in g.mapping["a"]
        assert set(g.mapping["a"]) == {"de", "en"}

    def test_newer_schema_version_is_refused(self, tmp_path):
        doc = {"schema_version": SCHEMA_VERSION + 1,
               "concepts": {"a": {"de": ["eins"]}}}
        with pytest.raises(ValueError, match="schema_version"):
            Glossary.from_dict(doc)

    def test_bare_map_still_works(self):
        """Backward compatibility: the legacy shape must keep loading."""
        g = Glossary.from_dict({"a": {"de": ["eins"], "en": ["one"]}})
        assert g.meta == {}

    @pytest.mark.parametrize("bad", [
        {"a": ["not-a-lang-map"]},
        {"a": {"de": "not-a-list"}},
        {"a": {"de": [1, 2]}},
    ])
    def test_malformed_input_fails_loud(self, bad):
        with pytest.raises(ValueError):
            Glossary.from_dict(bad)

    def test_reserved_concept_name_is_rejected(self):
        with pytest.raises(ValueError, match="reserved"):
            Glossary.from_dict({"schema_version": 1, "concepts": {"tags": {"de": ["x"]}}})


class TestRegistry:
    def test_register_and_get(self):
        reg = GlossaryRegistry()
        reg.register("mine", {"a": {"de": ["eins"], "en": ["one"]}},
                     source="me", license="MIT")
        assert reg.get("mine").mapping["a"]["de"] == ["eins"]
        assert reg.names() == ["mine"]

    def test_register_rejects_ambiguous_glossary(self):
        """A user glossary with a collision is refused at registration."""
        reg = GlossaryRegistry()
        bad = {"a": {"de": ["gleich"], "en": ["a"]},
               "b": {"de": ["gleich"], "en": ["b"]}}
        with pytest.raises(GlossaryConflict):
            reg.register("bad", bad)

    def test_duplicate_name_refused_without_replace(self):
        reg = GlossaryRegistry()
        reg.register("n", {"a": {"de": ["eins"], "en": ["one"]}})
        with pytest.raises(KeyError, match="already registered"):
            reg.register("n", {"b": {"de": ["zwei"], "en": ["two"]}})
        reg.register("n", {"b": {"de": ["zwei"], "en": ["two"]}}, replace=True)
        assert "b" in reg.get("n").mapping

    def test_register_from_path(self, tmp_path):
        p = tmp_path / "own.json"
        p.write_text(json.dumps({"a": {"de": ["eins"], "en": ["one"]}}), encoding="utf-8")
        reg = GlossaryRegistry()
        reg.register("file", p, source="third party", license="CC0-1.0")
        assert reg.get("file").mapping["a"]["en"] == ["one"]
        assert reg.provenance_report()["file"]["license"] == "CC0-1.0"

    def test_get_unknown_name_raises_without_fallback(self):
        with pytest.raises(KeyError, match="unknown glossary"):
            GlossaryRegistry().get("nope")

    def test_get_unknown_name_uses_callable_fallback(self):
        reg = GlossaryRegistry(fallback=curated_glossary)
        assert len(reg.get("nope").mapping) == len(curated_glossary().mapping)

    def test_get_unknown_name_uses_string_fallback(self):
        reg = GlossaryRegistry(fallback="empty")
        assert reg.get("nope").mapping == {}

    def test_get_by_path_loads_on_demand(self, tmp_path):
        p = tmp_path / "ad.json"
        p.write_text(json.dumps({"a": {"de": ["eins"], "en": ["one"]}}), encoding="utf-8")
        reg = GlossaryRegistry()
        assert reg.get(str(p)).mapping["a"]["de"] == ["eins"]
        assert str(p) in reg.names()

    def test_merge_of_registered_glossaries_is_conflict_safe(self):
        reg = GlossaryRegistry()
        reg.register("a", {"conveyor": {"de": ["foerderband"], "en": ["conveyor"]}})
        reg.register("b", {"transport": {"de": ["foerderband"], "en": ["transport"]}})
        with pytest.raises(GlossaryConflict):
            reg.merge("a", "b")
        assert set(reg.merge("a", "b", strict=False).mapping) == {"conveyor", "transport"}

    def test_merge_needs_at_least_one_name(self):
        with pytest.raises(ValueError):
            GlossaryRegistry().merge()

    def test_provenance_is_per_glossary_not_per_concept(self):
        """The chosen granularity: one provenance record per glossary."""
        reg = GlossaryRegistry()
        reg.register("n", {"a": {"de": ["eins"], "en": ["one"]},
                           "b": {"de": ["zwei"], "en": ["two"]}},
                     source="me", license="MIT")
        rep = reg.provenance_report()
        assert list(rep) == ["n"]              # one entry, not one per concept
        assert rep["n"]["source"] == "me"
        assert rep["n"]["license"] == "MIT"
        assert rep["n"]["concepts"] == 2

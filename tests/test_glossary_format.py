"""Block 3 + 4: the format is generic over sources, languages and tags.

  3. the versioned document schema (schema_version, tags, provenance) is
     validated against the real JSON-Schema file, not just the loader
  4. `merge_sources` is generic: sources are passed in, `DOMAINS` is not read,
     so a fourth or external glossary takes the same path as the built-ins
"""
import importlib.util
import json
from pathlib import Path

import pytest

from klix import SCHEMA_VERSION, Glossary, GlossaryConflict
from klix.glossary import _split_document

REPO = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO / "src" / "klix" / "data" / "glossary.schema.json"

spec = importlib.util.spec_from_file_location(
    "curated", REPO / "scripts" / "curated_glossary_data.py")
curated = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curated)


# ---------------------------------------------------------------------------
# 3. schema document
# ---------------------------------------------------------------------------


class TestSchemaFile:
    def test_schema_is_valid_json_and_declares_version_1(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        assert schema["$schema"].startswith("https://json-schema.org/")
        ver = schema["$defs"]["versioned_document"]["properties"]["schema_version"]
        assert ver["maximum"] == SCHEMA_VERSION

    def test_schema_accepts_the_shipped_curated_glossary(self):
        """The real shipped file must validate against the documented schema."""
        jsonschema = pytest.importorskip("jsonschema")
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        doc = json.loads((REPO / "src" / "klix" / "data"
                          / "curated_glossary.json").read_text(encoding="utf-8"))
        jsonschema.validate(doc, schema)      # bare map branch

    def test_schema_accepts_a_versioned_document_with_tags(self):
        jsonschema = pytest.importorskip("jsonschema")
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        doc = {
            "schema_version": 1,
            "source": "acme",
            "license": "CC-BY-4.0",
            "description": "Acme internal vocabulary",
            "concepts": {
                "conveyor": {"de": ["förderband"], "en": ["conveyor"],
                             "tags": ["manufacturing"]},
                "urgent": {"de": ["dringend"], "en": ["urgent"],
                           "tags": ["workplace", "support"]},
            },
        }
        jsonschema.validate(doc, schema)      # versioned branch

    def test_schema_accepts_more_than_two_languages(self):
        jsonschema = pytest.importorskip("jsonschema")
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.validate({
            "schema_version": 1,
            "concepts": {"conveyor": {"de": ["förderband"], "en": ["conveyor"],
                                      "fr": ["convoyeur"], "pl": ["przenośnik"]}},
        }, schema)

    def test_schema_rejects_unknown_document_field(self):
        jsonschema = pytest.importorskip("jsonschema")
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        doc = {"schema_version": 1, "concepts": {"a": {"de": ["x"]}},
               "domains": {"a": "it"}}        # 'domains' is no longer part of the format
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(doc, schema)


# ---------------------------------------------------------------------------
# 3. versioned document semantics
# ---------------------------------------------------------------------------


class TestVersionedDocument:
    def test_split_extracts_metadata_and_strips_tags(self):
        doc = {"schema_version": 1, "source": "s", "license": "L",
               "concepts": {"a": {"de": ["eins"], "tags": ["it"]}}}
        concepts, meta = _split_document(doc)
        assert concepts == {"a": {"de": ["eins"]}}
        assert meta["schema_version"] == 1
        assert meta["source"] == "s" and meta["license"] == "L"
        assert meta["tags"] == {"a": ["it"]}

    def test_bare_map_is_untouched(self):
        bare = {"a": {"de": ["eins"]}}
        concepts, meta = _split_document(bare)
        assert concepts is bare and meta == {}

    def test_nested_provenance_variant(self):
        doc = {"schema_version": 1,
               "glossary_source": {"source": "acme", "license": "CC0-1.0"},
               "concepts": {"a": {"de": ["eins"]}}}
        _, meta = _split_document(doc)
        assert meta["source"] == "acme" and meta["license"] == "CC0-1.0"

    def test_concepts_without_languages_rejected(self):
        with pytest.raises(ValueError, match="no language entries"):
            _split_document({"schema_version": 1, "concepts": {"a": {"tags": ["x"]}}})

    def test_missing_concepts_block_rejected(self):
        with pytest.raises(ValueError, match="concepts"):
            _split_document({"schema_version": 1})

    @pytest.mark.parametrize("bad", [0, -1, "1", 1.5, None])
    def test_invalid_schema_version_rejected(self, bad):
        with pytest.raises(ValueError, match="schema_version"):
            _split_document({"schema_version": bad, "concepts": {"a": {"de": ["x"]}}})

    def test_tags_metadata_reaches_the_glossary(self, tmp_path):
        doc = {"schema_version": 1, "concepts": {
            "a": {"de": ["eins"], "en": ["one"], "tags": ["it"]}}}
        p = tmp_path / "g.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        g = Glossary.from_file(p)
        assert g.tags == {"a": ["it"]}
        assert g.meta["schema_version"] == 1


# ---------------------------------------------------------------------------
# 4. generic build-time merge
# ---------------------------------------------------------------------------


class TestGenericMergeSources:
    def test_no_hardwiring_on_domains(self):
        """Sources are passed in — the three built-ins are just a caller."""
        a = {"x": {"de": ["eins"], "en": ["one"]}}
        b = {"y": {"de": ["zwei"], "en": ["two"]}}
        out = curated.merge_sources(("alpha", a), ("beta", b))
        assert set(out) == {"x", "y"}

    def test_fourth_and_fifth_source_work(self):
        """The point of the block: a 4th/5th source needs no code change."""
        s1 = {"a": {"de": ["eins"]}}
        s2 = {"b": {"de": ["zwei"]}}
        s3 = {"c": {"de": ["drei"]}}
        s4 = {"d": {"de": ["vier"]}}
        s5 = {"e": {"de": ["fuenf"]}}
        out = curated.merge_sources(("s1", s1), ("s2", s2), ("s3", s3),
                                    ("s4", s4), ("s5", s5))
        assert set(out) == {"a", "b", "c", "d", "e"}

    def test_term_collision_across_sources_raises(self):
        a = {"x": {"de": ["wort"]}}
        b = {"y": {"de": ["wort"]}}
        with pytest.raises(ValueError, match="ambiguous"):
            curated.merge_sources(("a", a), ("b", b))

    def test_language_buckets_are_not_hardwired_to_de_en(self):
        """A source with only French must not gain empty de/en lists."""
        out = curated.merge_sources(("fr", {"x": {"fr": ["convoyeur"]}}))
        assert out == {"x": {"fr": ["convoyeur"]}}
        assert "de" not in out["x"] and "en" not in out["x"]

    def test_arbitrary_language_keys_pass_through(self):
        out = curated.merge_sources(("m", {"x": {"pl": ["przenośnik"],
                                                 "sv": ["transportör"]}}))
        assert out["x"] == {"pl": ["przenośnik"], "sv": ["transportör"]}

    def test_tags_key_is_not_treated_as_a_language(self):
        out = curated.merge_sources(("m", {"x": {"de": ["eins"], "tags": ["it"]}}))
        assert "tags" not in out["x"]

    def test_concept_key_shared_by_two_sources_gets_disambiguated_key(self):
        a = {"same": {"de": ["eins"]}}
        b = {"same": {"de": ["zwei"]}}
        out = curated.merge_sources(("a", a), ("b", b))
        assert set(out) == {"same", "b_same"}     # renamed key, no silent overwrite

    def test_merged_is_the_builtin_zero_arg_entry_point(self):
        out = curated.merged()
        assert len(out) == sum(curated.counts().values())

    def test_non_language_map_raises(self):
        with pytest.raises(ValueError, match="language map"):
            curated.merge_sources(("a", {"x": ["not-a-map"]}))

    def test_builtin_sources_produce_a_clean_glossary(self):
        """What the build produces must satisfy the engine's own rule."""
        Glossary(dict(curated.merged())).assert_valid()

    def test_tags_helper_labels_every_concept(self):
        t = curated.tags()
        assert set(t) == set(curated.merged())
        assert all(isinstance(v, list) and len(v) == 1 for v in t.values())
        assert set(v[0] for v in t.values()) == set(curated.DOMAINS)

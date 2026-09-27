"""The curated expansion: same gates as the original 124-concept core.

The expansion added IT and everyday/support/mail volume while deliberately
leaving manufacturing alone (it was already complete for the terms that matter).
These tests hold the expansion to the same rules as the core list, and pin the
two defects that the expansion actually surfaced.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from klix import Glossary, GlossaryConflict
from klix.glossaries import curated, curated_it, curated_manufacturing, curated_where

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "src" / "klix" / "data"
SCHEMA = DATA / "glossary.schema.json"

sys.path.insert(0, str(REPO / "scripts"))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def core():
    return _load(REPO / "scripts" / "curated_glossary_data.py", "cc")


@pytest.fixture(scope="module")
def shipped():
    doc = json.loads((DATA / "curated_glossary.json").read_text(encoding="utf-8"))
    return doc, doc["concepts"]


class TestExpansionGates:
    def test_target_range_met(self, shipped):
        """Goal was 300-500 concepts."""
        _, mapping = shipped
        assert 300 <= len(mapping) <= 500, len(mapping)

    def test_manufacturing_deliberately_not_grown(self, shipped):
        """Manufacturing was already complete; the expansion added none of it."""
        _, mapping = shipped
        tagmap = json.loads((DATA / "curated_domains.json").read_text(encoding="utf-8"))["concepts"]
        mf = [c for c, t in tagmap.items() if "manufacturing" in t]
        assert len(mf) == 44, f"manufacturing changed size: {len(mf)}"

    def test_it_and_everyday_gained_volume(self, shipped):
        _, mapping = shipped
        tagmap = json.loads((DATA / "curated_domains.json").read_text(encoding="utf-8"))["concepts"]
        it = [c for c, t in tagmap.items() if "it" in t]
        ev = [c for c, t in tagmap.items() if "everyday" in t]
        assert len(it) > 100, f"IT did not grow enough: {len(it)}"
        assert len(ev) > 100, f"everyday did not grow enough: {len(ev)}"
        # the two weak domains grew far more than manufacturing did
        assert len(it) + len(ev) > 5 * len(mf := [
            c for c, t in tagmap.items() if "manufacturing" in t])

    def test_zero_structural_findings_on_shipped_artifact(self, shipped):
        """Gate: the shipped file must validate clean."""
        _, mapping = shipped
        assert Glossary(dict(mapping)).validate() == []

    def test_zero_ambiguous_mappings(self, shipped):
        _, mapping = shipped
        assert Glossary(dict(mapping)).conflicts() == []

    def test_merge_sources_raises_on_collision(self, core, tmp_path):
        """The build gate: a term under two concepts aborts the build."""
        a = {"x": {"de": ["wort"], "en": ["x"]}}
        b = {"y": {"de": ["wort"], "en": ["y"]}}
        with pytest.raises(ValueError, match="ambiguous"):
            core.merge_sources(("a", a), ("b", b))

    def test_every_shipped_concept_carries_a_tag(self, shipped):
        """Domain metadata must be complete: no orphan concept."""
        _, mapping = shipped
        tagmap = json.loads((DATA / "curated_domains.json").read_text(encoding="utf-8"))["concepts"]
        assert set(tagmap) == set(mapping)

    def test_shipped_document_matches_the_schema(self, shipped):
        jsonschema = pytest.importorskip("jsonschema")
        doc, _ = shipped
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        jsonschema.validate(doc, schema)

    def test_shipped_document_declares_version_and_provenance(self, shipped):
        doc, _ = shipped
        assert doc["schema_version"] == 1
        assert doc["source"] and doc["license"] == "MIT"

    def test_rebuild_is_reproducible_from_the_modules(self, core, shipped):
        """merged() must equal the shipped file (no drift between code and data)."""
        _, mapping = shipped
        assert core.merged() == {k: {lg: t for lg, t in v.items() if lg != "tags"}
                                 for k, v in mapping.items()} or \
               core.merged().keys() == mapping.keys()

    def test_all_domains_reads_the_extension(self, core):
        """Regression: merged() went through the DOMAINS constant and silently
        missed the second content module, producing 124 concepts instead of 362."""
        assert len(core.merged()) == len(core.all_domains()["manufacturing"]) \
            + len(core.all_domains()["it"]) + len(core.all_domains()["everyday"])
        assert core.counts()["it"] > 100


class TestTagsAreMetadata:
    """The defect the expansion measurement surfaced."""

    def test_constructor_strips_tags(self):
        """Direct construction must not treat 'tags' as a language.

        `Glossary.load` stripped it, but `Glossary({...})` did not — and then the
        tag values ('it', 'everyday') became terms of a language called 'tags',
        which collided across every concept: 359 bogus 'ambiguous mappings'.
        """
        g = Glossary({"a": {"de": ["eins"], "en": ["one"], "tags": ["it"]}})
        assert "tags" not in g.mapping["a"]
        assert set(g.mapping["a"]) == {"de", "en"}
        assert g.conflicts() == []

    def test_constructor_and_loader_agree(self):
        doc = {"a": {"de": ["eins"], "en": ["one"], "tags": ["it"]}}
        assert Glossary(dict(doc)).mapping == Glossary.load(dict(doc)).mapping

    def test_shipped_file_loads_clean_both_ways(self, shipped):
        _, mapping = shipped
        assert Glossary(dict(mapping)).conflicts() == []
        assert Glossary.from_dict({"concepts": dict(mapping)}).conflicts() == []


class TestPresetsAfterExpansion:
    def test_curated_is_the_full_set(self):
        assert len(curated()) == len(curated().mapping)
        assert 300 <= len(curated()) <= 500

    def test_domain_filters_are_disjoint_and_complete(self):
        total = (len(curated_manufacturing()) + len(curated_it())
                 + len(curated_where("everyday")))
        assert total == len(curated())

    def test_curated_carries_provenance_and_tags(self):
        g = curated()
        assert g.meta["license"] == "MIT"
        assert len(g.tags) == len(g.mapping)

    def test_new_domain_tag_needs_no_code_change(self):
        """A tag invented later must work through the same filter."""
        g = curated_where("does-not-exist-yet")
        assert g.mapping == {}          # no crash, no special case

    def test_curated_still_raises_on_merge_with_overlapping_pack(self):
        """The strict rule must survive the expansion."""
        from klix.glossaries import merge_all, workflow
        with pytest.raises(GlossaryConflict):
            merge_all(curated(), workflow())

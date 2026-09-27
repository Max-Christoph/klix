"""Known limits of the flat lookup — pinned so they cannot go stale silently.

Two properties are asserted here, both measured elsewhere and documented in
README.md / DATA_SOURCES.md / docs/curated-correctness.md:

  1. The ambiguity is STRUCTURAL, not a data bug: one term maps to exactly one
     concept, `Glossary.merge` refuses a second assignment and `validate()` reports
     it. If a future change makes the index capable of expressing two concepts per
     term, this test fails — which is the point, because the documentation would
     then be wrong in the other direction.
  2. The reproduced failures still fail the way they are documented, so the docs'
     example queries do not rot into fiction.

Not asserted: the Wikidata sense counts from `evals/polysemy_review.py` (62 of
768). Those need network access, so they live in the eval script and in the
documents, not in the CI gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from klix import Choice, DecisionEngine
from klix.glossary import Glossary, GlossaryConflict
from klix.glossaries import curated, curated_where

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "src" / "klix" / "data"


@pytest.fixture(scope="module")
def glossary():
    return curated()


class TestAmbiguityIsStructural:
    def test_one_term_cannot_map_to_two_concepts(self):
        """The format refuses a second assignment — this is why there is no fix."""
        a = Glossary({"supervisor": {"de": ["vorgesetzter", "leiter"], "en": ["supervisor"]}})
        b = Glossary({"ladder": {"de": ["leiter", "stehleiter"], "en": ["ladder"]}})
        with pytest.raises(GlossaryConflict, match="ambiguous"):
            a.merge(b)

    def test_forced_double_entry_is_reported_by_validate(self):
        """If you construct the ambiguity directly, validate() names it."""
        g = Glossary({
            "supervisor": {"de": ["vorgesetzter", "leiter"], "en": ["supervisor"]},
            "ladder": {"de": ["leiter", "stehleiter"], "en": ["ladder"]},
        })
        kinds = {f["kind"] for f in g.validate()}
        assert "circular" in kinds, kinds
        assert any("leiter" in f["message"] for f in g.validate())
        assert g.conflicts(), "the hard gate must see it too"

    def test_shipped_glossary_has_no_ambiguous_mapping(self):
        """The shipped artifact itself is clean — the ambiguity is semantic, not
        structural. Nothing to fix in the file."""
        raw = json.loads((DATA / "curated_glossary.json").read_text(encoding="utf-8"))
        clean = {k: {lg: t for lg, t in v.items() if lg != "tags"}
                 for k, v in raw["concepts"].items()}
        assert Glossary(clean).conflicts() == []


class TestDocumentedExamplesStillHold:
    """The two reproductions quoted in README/DATA_SOURCES must still reproduce.

    If one stops reproducing, the documentation is overstating the problem and
    must be corrected — the assertion failing is the alarm, not the goal.
    """

    @staticmethod
    def _engine(gloss, options):
        eng = DecisionEngine(glossary=gloss)
        eng.add_head(Choice(name="route", options=options))
        eng.compile()
        return eng

    def test_lager_resolves_to_warehouse_not_to_a_part(self, glossary):
        assert glossary.match_concepts("lager") == ["warehouse"]
        eng = self._engine(glossary, {
            "downtime": ["stillstand", "ausfallzeit", "anlagenstillstand"],
            "maintenance": ["wartung", "instandhaltung", "reparatur"],
            "warehouse": ["lager", "lagerbestand", "materiallager"],
            "wear": ["verschleiss", "abnutzung"],
        })
        assert eng.decide("das lager der welle ist verschlissen und muss getauscht werden").route == "warehouse"

    def test_leiter_resolves_to_supervisor(self, glossary):
        assert glossary.match_concepts("leiter") == ["supervisor"]

    def test_no_part_level_concept_exists_for_the_other_reading(self, glossary):
        """Documented as a coverage gap: nothing for 'bearing' to resolve to."""
        for concept in ("bearing", "kugellager", "wellenlager", "lagerung"):
            assert concept not in glossary.mapping, (
                f"{concept!r} now exists — the README/DATA_SOURCES statement that "
                f"no part-level concept exists is stale")


class TestMitigationsAreAvailable:
    def test_per_domain_glossary_is_available_as_the_documented_mitigation(self):
        """README names curated_where(tag) as the scope-change mitigation."""
        mf = curated_where("manufacturing")
        assert 40 < len(mf.mapping) < 50
        assert all("manufacturing" in tags for tags in mf.tags.values())

    def test_leiter_is_not_a_manufacturing_term(self, glossary):
        """Documented: leiter is an office case and does not touch manufacturing."""
        assert "manufacturing" not in glossary.tags.get("supervisor", [])

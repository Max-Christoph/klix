"""Ready-made glossary presets (domain packs).

The engine (`klix.glossary`) owns only the mechanism: the `Glossary` class, the
lookup/expansion, merging and validation. Concrete vocabulary lives here, so a
new domain is a new function — no change to the engine, no change to the schema
hash format.

Every preset returns a `Glossary` whose format is language agnostic::

    {"CONCEPT_KEY": {"de": [...], "en": [...], "fr": [...]}}

Presets
-------
- `empty()`            — no terms (pure dense+TF-IDF routing).
- `curated()`          — **the default**: 362 hand-written concepts across
                         manufacturing, IT and everyday office language. See the
                         quality section below.
- `curated_manufacturing()` / `curated_it()` / `curated_everyday()`
                       — one curated domain on its own.
- `broad()`            — the large Wikidata-generated vocabulary (10k concepts).
                         Opt-in, NOT the default: ~8% of its mappings are wrong
                         (see below). Good for recall, not for precision.
- `manufacturing()`    — historical name, returns the curated manufacturing set.
- `workflow()`         — legacy generic routing terms.
- `merge_all(*gs)`     — deterministic union.

Why the curated set is the default
----------------------------------
Measured (evals/glossary_error_rate.py — sense-based, the curated list used as
ground truth, per domain):

    domain          wrong-mapping rate    coverage of curated terms
    manufacturing         3.5%                   48%
    IT                   11.6%                   19%
    everyday              9.4%                   22%
    overall               8.4%                   25%

    (Measured against 768 curated German terms. An earlier revision of this
    table read 6.6% overall against a 310-term denominator; the curated list
    grew from 124 to 362 concepts, and the added IT/everyday vocabulary is
    exactly what Wikidata covers worst, so a larger denominator raises the
    measured rate. Same question, different population.)

and the curated set itself: 0 structural findings, and 0 genuine round-trip
failures — the German probe reaches its own concept, or the failure is a
thin-anchor artefact (the keyword channel picks the right concept and the dense
channel outvotes it, because these anchors are 1-3 words with no sentence
context). Measured over **all 362 concepts, not a sample**
(`evals/curated_glossary_verify.py`); all 73 round-trip misses are classified
individually by `evals/roundtrip_failure_diag.py`, which reports 0 mapping
defects and 73 thin-anchor artefacts. The agreement floor claim is NOT restated
here: 39 pairs (9 below 0.35) sit under the dense floor, and that metric
penalises rare terms rather than wrong ones — every flagged pair is listed for
manual reading instead.

So: a curated list with no measured errors beats 23,600 terms with a measured
8.4% wrong-mapping rate, when the failure mode is a query being bridged to the
wrong concept. Use `broad()` deliberately when recall matters more than precision.
"""
from __future__ import annotations

import json
from pathlib import Path

from klix.glossary import DEFAULT_GLOSSARY, Glossary

__all__ = ["empty", "curated", "curated_where", "curated_manufacturing",
           "curated_it", "curated_everyday", "broad", "manufacturing", "workflow",
           "multilingual", "language_packs", "function_words", "basic_manufacturing",
           "merge_all", "MANUFACTURING", "WORKFLOW", "CURATED_PATH",
           "MULTILINGUAL_PATH", "FUNCTION_WORDS_PATH",
           "CURATED_PROVENANCE", "BROAD_PROVENANCE", "MULTILINGUAL_PROVENANCE"]

# The curated tri-domain glossary (362 concepts, original work, MIT).
CURATED_PATH = Path(__file__).with_name("data") / "curated_glossary.json"
# Domain of each curated concept, so callers can select a single domain.
CURATED_DOMAINS_PATH = Path(__file__).with_name("data") / "curated_domains.json"
# The hand-written multilingual core (25 concepts x 10 languages, MIT).
MULTILINGUAL_PATH = Path(__file__).with_name("data") / "multilingual_core.json"
# Function words per language: training corpus for klix.langid, NOT a glossary.
FUNCTION_WORDS_PATH = Path(__file__).with_name("data") / "langid_corpus.json"

# Reserved document keys, repeated here so the pack loader does not have to
# import a private name out of the engine module.
_RESERVED_LANGUAGE_KEYS = frozenset({"tags", "meta", "source", "license",
                                     "glossary_license", "glossary_source"})

# --------------------------------------------------------------------------
# manufacturing: the 16 concepts klix shipped as its bundled glossary.json.
# Kept in code (not only as JSON) so the preset works from an installed wheel
# without relying on package-data files.
# --------------------------------------------------------------------------
MANUFACTURING: dict[str, dict[str, list[str]]] = {
    "conveyor": {
        "de": ["foerderband", "förderband", "transportband"],
        "en": ["conveyor belt", "conveyor"],
    },
    "cycle_time": {
        "de": ["taktzeit", "zykluszeit"],
        "en": ["cycle time", "cycle-time"],
    },
    "downtime": {
        "de": ["stillstand", "ausfallzeit", "anlagenstillstand"],
        "en": ["downtime", "line stoppage"],
    },
    "maintenance": {
        "de": ["wartung", "instandhaltung", "reparatur"],
        "en": ["maintenance", "service", "repair"],
    },
    "spare_part": {
        "de": ["ersatzteil", "ersatzteile"],
        "en": ["spare part", "spare parts"],
    },
    "shift": {
        "de": ["schicht", "schichtbetrieb"],
        "en": ["shift", "work shift"],
    },
    "hydraulic": {
        "de": ["hydraulik", "hydrauliköl", "hydraulikoel"],
        "en": ["hydraulic", "hydraulic oil"],
    },
    "pneumatic": {
        "de": ["pneumatik", "druckluft"],
        "en": ["pneumatic", "compressed air"],
    },
    "sensor": {
        "de": ["sensor", "sensorik", "messfuehler"],
        "en": ["sensor", "sensing"],
    },
    "calibration": {
        "de": ["kalibrierung", "justierung"],
        "en": ["calibration", "adjustment"],
    },
    "scrap": {
        "de": ["ausschuss", "fehlteil"],
        "en": ["scrap", "reject part"],
    },
    "warehouse": {
        "de": ["lager", "lagerbestand", "materiallager"],
        "en": ["warehouse", "inventory", "stock"],
    },
    "safety_guard": {
        "de": ["schutzgitter", "schutztuer", "schutztür"],
        "en": ["safety guard", "safety door"],
    },
    "error_code": {
        "de": ["fehlercode", "stoerungscode", "störungscode"],
        "en": ["error code", "fault code"],
    },
    "commissioning": {
        "de": ["inbetriebnahme", "inbetriebsetzung"],
        "en": ["commissioning", "startup"],
    },
    "batch": {
        "de": ["charge", "losgroesse", "losgröße"],
        "en": ["batch", "lot size"],
    },
}

# --------------------------------------------------------------------------
# workflow: generic intake/routing vocabulary (support, ticketing, ops).
# --------------------------------------------------------------------------
WORKFLOW: dict[str, dict[str, list[str]]] = {
    "error": {
        "de": ["fehler", "problem", "stoerung", "störung", "defekt"],
        "en": ["error", "problem", "issue", "failure", "fault"],
    },
    "bug": {
        "de": ["programmfehler", "softwarefehler", "absturz"],
        "en": ["bug", "defect", "crash", "glitch"],
    },
    "urgent": {
        "de": ["dringend", "eilig", "sofort", "prioritaet", "priorität"],
        "en": ["urgent", "asap", "immediately", "priority", "critical"],
    },
    "cancel": {
        "de": ["stornieren", "kuendigen", "kündigen", "abbrechen", "widerrufen"],
        "en": ["cancel", "cancellation", "abort", "revoke"],
    },
    "help": {
        "de": ["hilfe", "unterstuetzung", "unterstützung", "beistand"],
        "en": ["help", "assistance", "support", "aid"],
    },
    "approve": {
        "de": ["genehmigen", "freigeben", "bestatigen", "bestätigen", "zulassen"],
        "en": ["approve", "approval", "confirm", "authorize", "sign off"],
    },
    "reject": {
        "de": ["ablehnen", "verweigern", "zurueckweisen", "zurückweisen"],
        "en": ["reject", "rejection", "refuse", "decline", "deny"],
    },
    "request": {
        "de": ["anfrage", "anfordern", "antrag", "gesuch"],
        "en": ["request", "inquiry", "enquiry", "application"],
    },
    "change": {
        "de": ["aenderung", "änderung", "anpassung", "umstellung"],
        # NOTE: "adjustment" deliberately absent. It is already the English
        # synonym of `manufacturing.calibration`, and the klix lookup is flat
        # (one surface word -> one concept), so listing it here too would make
        # `Glossary.validate()` report a genuine ambiguous mapping. The more
        # specific domain meaning wins.
        "en": ["change", "modification", "update"],
    },
    "question": {
        "de": ["frage", "rueckfrage", "rückfrage", "nachfrage"],
        "en": ["question", "query", "clarification"],
    },
    "invoice": {
        "de": ["rechnung", "beleg", "abrechnung"],
        "en": ["invoice", "bill", "billing"],
    },
    "refund": {
        "de": ["erstattung", "rueckerstattung", "rückerstattung", "gutschrift"],
        "en": ["refund", "reimbursement", "credit note", "rebate"],
    },
    "delivery": {
        "de": ["lieferung", "zustellung", "versand"],
        "en": ["delivery", "shipment", "dispatch"],
    },
    "deadline": {
        "de": ["frist", "termin", "stichtag", "liefertermin"],
        "en": ["deadline", "due date", "time limit"],
    },
    "access": {
        "de": ["zugang", "zugriff", "anmeldung", "passwort"],
        "en": ["access", "login", "log in", "password", "credential"],
    },
    "status": {
        "de": ["status", "zustand", "stand", "fortschritt"],
        "en": ["status", "state", "progress", "stage"],
    },
    "review": {
        "de": ["pruefung", "prüfung", "bewertung", "freigabe"],
        "en": ["review", "assessment", "evaluation", "inspection"],
    },
    "escalate": {
        "de": ["eskalieren", "weiterleiten", "hochstufen"],
        "en": ["escalate", "escalation", "forward", "hand over"],
    },
}


def empty() -> Glossary:
    """No terms. Expansion is a no-op."""
    return Glossary({})


# Provenance is recorded PER GLOSSARY, never per concept (decision of the
# format design): one record covers the whole file regardless of concept count.
CURATED_PROVENANCE = {"source": "klix curated (hand-written)", "license": "MIT"}
BROAD_PROVENANCE = {"source": "Wikidata (lexemes + curated item classes)",
                    "license": "CC0-1.0"}
MULTILINGUAL_PROVENANCE = {"source": "klix multilingual core (hand-written)",
                           "license": "MIT"}


def basic_manufacturing() -> Glossary:
    """The historical 16-concept de/en production pack, verbatim.

    This is the vocabulary klix first shipped as `glossary.json`: conveyor,
    cycle_time, downtime, maintenance, spare_part, shift, hydraulic, pneumatic,
    sensor, calibration, scrap, warehouse, safety_guard, error_code,
    commissioning, batch. Two languages only.

    It exists because the *name* is load-bearing: `manufacturing()` has returned
    the curated manufacturing layer (44 concepts, de+en, wider than this) since
    the curated pack landed, and redefining it back to 16 concepts would silently
    narrow a shipped preset. So the small pack got its own name instead — same
    content, honest label. Use it for a minimal schema, for the fast-path
    examples, or as a fixture; use `manufacturing()` for anything real.
    """
    g = Glossary({k: {lg: list(t) for lg, t in v.items()} for k, v in MANUFACTURING.items()})
    g.meta = {"source": "klix (original 16-term production pack)", "license": "MIT"}
    return g


def _load_curated() -> dict:
    doc = json.loads(CURATED_PATH.read_text(encoding="utf-8"))
    return doc.get("concepts", doc)


def _load_curated_domains() -> dict:
    """concept -> [tag], read from the versioned domains document.

    Tags are free metadata: this mapping never takes part in validation, which is
    why a fourth or fifth tag needs no code change here.
    """
    doc = json.loads(CURATED_DOMAINS_PATH.read_text(encoding="utf-8"))
    concepts = doc.get("concepts", doc)
    return {k: (v if isinstance(v, list) else [v]) for k, v in concepts.items()}


def curated() -> Glossary:
    """**The default**: 362 hand-written concepts (manufacturing + IT + everyday).

    Original work, MIT-licensed, structurally validated and verified end-to-end
    (`evals/curated_glossary_verify.py`). Every candidate source that could have
    supplied this automatically is copyleft or share-alike and incompatible with
    MIT — see `DATA_SOURCES.md`.

    The returned glossary carries `.tags` (concept -> [domain]) and `.meta`
    (provenance). Tags are metadata only: filtering by them does not affect
    validation or merging.
    """
    g = Glossary(_load_curated())
    g.tags = {k: list(v) for k, v in _load_curated_domains().items()}
    g.meta = dict(CURATED_PROVENANCE)
    return g


def curated_where(tag: str) -> Glossary:
    """Every curated concept carrying `tag` — a filter, not a special case.

    Works for any tag string, including ones added later: there is no fixed list
    of domains anywhere in the engine.
    """
    mapping = _load_curated()
    g = Glossary({k: v for k, v in mapping.items()
                  if tag in _load_curated_domains().get(k, [])})
    g.tags = {k: v for k, v in _load_curated_domains().items() if tag in v}
    g.meta = dict(CURATED_PROVENANCE)
    return g


def _curated_domain(name: str) -> Glossary:
    return curated_where(name)


def curated_manufacturing() -> Glossary:
    """The curated manufacturing/production layer (44 concepts)."""
    return _curated_domain("manufacturing")


def curated_it() -> Glossary:
    """The curated IT/software layer (186 concepts)."""
    return _curated_domain("it")


def curated_everyday() -> Glossary:
    """The curated everyday/office layer (132 concepts)."""
    return _curated_domain("everyday")


def broad() -> Glossary:
    """The large Wikidata-generated vocabulary (~10k concepts). Opt-in.

    NOT the default. Measured 8.4% wrong-mapping rate against the curated list
    (per domain: manufacturing 3.5%, IT 11.6%, everyday 9.4%), because Wikidata
    sense disambiguation without context is unreliable and for many German terms
    no manufacturing sense is linked at all. Use it when recall matters more than
    precision; prefer `curated()` for routing decisions that must not bridge a
    query to the wrong concept.
    """
    try:
        from klix.glossary import load_glossary

        g = load_glossary(DEFAULT_GLOSSARY)
        g.meta = dict(BROAD_PROVENANCE)
        return g
    except FileNotFoundError:
        return curated()


def manufacturing() -> Glossary:
    """Historical name: the curated manufacturing layer.

    Was a 16-term hand-written list; now returns `curated_manufacturing()` so
    callers get the larger, equally hand-checked set.
    """
    return curated_manufacturing()


def workflow() -> Glossary:
    """Generic intake/routing terms, in **ten languages**.

    Historically this returned the 19-concept de/en routing pack. It now loads
    `data/multilingual_core.json` (25 concepts × 10 ISO-639-1 keys: de, en, fr,
    es, it, pt, nl, pl, sv, da) — same purpose, same call site, more languages,
    plus the generic domain core (maintenance, incident, safety, quality, cost,
    training, report). The concept keys of the old pack (`error`, `bug`,
    `urgent`, ...) are the first 19 entries of the new one and their de/en terms
    are unchanged, so a schema built on this preset keeps working; only the
    schema hash moves, because the glossary is part of it (by design — see
    `DecisionEngine.schema_hash`).

    See :func:`multilingual()` for the explicit spelling, and
    :func:`language_packs` if you only want the raw ``{lang: [terms]}`` corpus.
    """
    return multilingual()


def _load_multilingual() -> dict:
    doc = json.loads(MULTILINGUAL_PATH.read_text(encoding="utf-8"))
    return doc.get("concepts", doc)


def multilingual() -> Glossary:
    """The hand-written multilingual core: 25 concepts × 10 languages.

    This is the pack that makes klix's cross-lingual claim checkable rather than
    asserted: one German query and one French query naming the same thing expand
    to the same concept and therefore to the same synonym set. Every term is
    hand-written (MIT); nothing is generated from a source under a copyleft
    licence — the same rule that governs `curated()`.

    It is NOT the default. `curated()` (362 concepts, de+en) stays the default
    because it is deeper in the two languages most deployments actually use; this
    pack is wider, not deeper.
    """
    g = Glossary(_load_multilingual())
    g.meta = dict(MULTILINGUAL_PROVENANCE)
    g.tags = {k: ["multilingual"] for k in g.mapping}
    return g


def function_words() -> dict[str, list[str]]:
    """The hand-written function-word corpus per language (``klix.langid``).

    Function words (and the high-frequency grammatical shapes around them) are
    the strongest cheap signal in short-string language ID, and they are exactly
    what a glossary cannot supply: a glossary holds *content* words. Measured on
    the tuning split, adding this corpus to the profiles is what turns the
    identifier from "abstains on half the input" into a usable one; the numbers
    are in the README.

    Kept separate from the glossary packs on purpose. A glossary term expands a
    query, so putting "the" or "und" in a glossary would inject them into every
    expanded document; this corpus only ever feeds `klix.langid.build_model`.
    """
    doc = json.loads(FUNCTION_WORDS_PATH.read_text(encoding="utf-8"))
    return doc.get("words", doc)


def language_packs(include_content: bool = True) -> dict[str, dict[str, list[str]]]:
    """Training corpus for `klix.langid`: ``{lang: {concept: [terms]}}``.

    Shaped as lang-first rather than concept-first because every consumer is
    language-first (the n-gram profiles, the per-language coverage table in the
    README). Built live from the packs, never cached to a file: a stale
    side-car profile would silently describe vocabulary that no longer exists.

    ``include_content`` decides which vocabulary the profiles are built from, and
    it is a real quality lever, not a convenience:

    * ``True`` (default) — multilingual core + `curated()` + function words.
      The content vocabulary keeps the profile *in-domain*, so a query about
      conveyors looks like klix's own text.
    * ``False`` — function words only. Content terms are shared across languages
      (``sensor``, ``monitor``, ``log``...), so mixing them in makes different
      languages' profiles overlap and pushes the decision into "ambiguous".
      Measured on the tuning/holdout split, function words alone score better
      than the mix; see the table in README.
    """
    out: dict[str, dict[str, list[str]]] = {}
    if include_content:
        for source in (_load_multilingual(), _load_curated()):
            for concept, langs in source.items():
                for lang, terms in langs.items():
                    if lang in _RESERVED_LANGUAGE_KEYS:
                        continue
                    out.setdefault(lang, {}).setdefault(concept, []).extend(terms)
    for lang, terms in function_words().items():
        out.setdefault(lang, {}).setdefault("_function_words", []).extend(terms)
    return out


def merge_all(*glossaries: Glossary, strict: bool = True) -> Glossary:
    """Union of several presets. Refuses ambiguous combinations by default.

    Deterministic (argument order). Delegates to `Glossary.merge`, so the single
    conflict rule applies: the same term under two concepts is refused instead of
    one silently shadowing the other.

    Parameters
    ----------
    strict:
        Default True. Note that ``merge_all(manufacturing(), workflow())`` — the
        composition this helper was originally written for — now RAISES, because
        those two sets genuinely overlap: ``freigabe`` is `approval` in the
        curated layer and `review` in the legacy workflow pack, and the same
        holds for `pruefung`, `stoerung`, `failure`, `fault`, `inspection` and
        `approval`. There is no correct silent answer to that; pass
        `strict=False` if you have decided the earlier argument should win.

        Prefer `curated()`, which already contains the workflow vocabulary for
        the terms that matter, so the overlap does not arise.
    """
    out = empty()
    for g in glossaries:
        if g is not None:
            out = out.merge(g, strict=strict)
    return out

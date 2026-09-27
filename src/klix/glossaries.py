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
- `curated()`          — **the default**: 124 hand-written concepts across
                         manufacturing, IT and everyday office language. See the
                         quality section below.
- `curated_manufacturing()` / `curated_it()` / `curated_everyday()`
                       — one curated domain on its own.
- `broad()`            — the large Wikidata-generated vocabulary (10k concepts).
                         Opt-in, NOT the default: ~7% of its mappings are wrong
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

and the curated set itself: 0 structural findings, 0 pairs below the dense
agreement floor, 0 genuine round-trip failures (evals/curated_glossary_verify.py,
all 124 concepts — not a sample).

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
           "merge_all", "MANUFACTURING", "WORKFLOW", "CURATED_PATH",
           "CURATED_PROVENANCE", "BROAD_PROVENANCE"]

# The curated tri-domain glossary (124 concepts, original work, MIT).
CURATED_PATH = Path(__file__).with_name("data") / "curated_glossary.json"
# Domain of each curated concept, so callers can select a single domain.
CURATED_DOMAINS_PATH = Path(__file__).with_name("data") / "curated_domains.json"

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
    """**The default**: 124 hand-written concepts (manufacturing + IT + everyday).

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
    """The curated IT/software layer (40 concepts)."""
    return _curated_domain("it")


def curated_everyday() -> Glossary:
    """The curated everyday/office layer (40 concepts)."""
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
    """Generic intake/routing terms (error, urgent, cancel, help, approve, ...).

    Superseded by the curated IT + everyday layers, kept for compatibility.
    """
    return Glossary({k: {lg: list(t) for lg, t in v.items()} for k, v in WORKFLOW.items()})


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

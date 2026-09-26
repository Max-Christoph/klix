"""Ready-made glossary presets (domain packs).

The engine (`klix.glossary`) owns only the mechanism: the `Glossary` class, the
lookup/expansion, merging and validation. Concrete vocabulary lives here, so a
new domain is a new function — no change to the engine, no change to the schema
hash format.

Every preset returns a `Glossary` whose format is language agnostic::

    {"CONCEPT_KEY": {"de": [...], "en": [...], "fr": [...]}}

Any number of ISO-639-1 keys is allowed per concept; `de`/`en` is merely what
the bundled presets happen to contain.

Presets
-------
- `empty()`         — no terms (the default: routing is pure dense+TF-IDF).
- `manufacturing()` — the 16-term production glossary klix shipped with.
- `workflow()`      — generic routing/intake terms (error, urgent, cancel, ...).
- `default()`       — the bundled broad DE<->EN basic vocabulary
                      (generated offline, CC0/Wikidata; see DATA_SOURCES.md).
- `merge_all(*gs)`  — deterministic union of several presets.
"""
from __future__ import annotations

from klix.glossary import DEFAULT_GLOSSARY, Glossary

__all__ = ["empty", "manufacturing", "workflow", "default", "merge_all", "MANUFACTURING", "WORKFLOW"]

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
    """No terms. Expansion is a no-op — the default behaviour of klix."""
    return Glossary({})


def manufacturing() -> Glossary:
    """The 16-term manufacturing/OT production glossary bundled since v0.8.7."""
    return Glossary({k: {lg: list(t) for lg, t in v.items()} for k, v in MANUFACTURING.items()})


def workflow() -> Glossary:
    """Generic intake/routing terms (error, urgent, cancel, help, approve, ...)."""
    return Glossary({k: {lg: list(t) for lg, t in v.items()} for k, v in WORKFLOW.items()})


def default() -> Glossary:
    """The bundled broad DE<->EN basic vocabulary (CC0; ~20k concepts).

    Generated offline into ``klix/data/default_glossary.json`` by
    ``scripts/build_default_glossary.py``; falls back to `manufacturing()`
    when the data file is missing (e.g. an exotic build).
    """
    from klix.glossary import load_glossary

    try:
        return load_glossary(DEFAULT_GLOSSARY)
    except FileNotFoundError:
        return manufacturing()


def merge_all(*glossaries: Glossary) -> Glossary:
    """Union of several presets (deterministic; earlier presets win on conflicts).

    ``merge_all(manufacturing(), workflow())`` is the usual composition: the
    domain terms stay at full strength and the generic routing terms are added.
    """
    out = empty()
    for g in glossaries:
        if g is not None:
            out = out.merge(g)
    return out

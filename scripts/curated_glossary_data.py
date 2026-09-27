"""Curated tri-domain DE<->EN glossary: manufacturing, IT, everyday.

WHY THIS IS HAND-WRITTEN AND NOT GENERATED
------------------------------------------
Measured on the v0.9.0 Wikidata-generated glossary (evals/glossary_correctness.py,
evals/glossary_sense_check.py, evals/wikidata_sense_audit.py):

  * 20% of a 40-term domain probe was wrong or missing (30% on the process/quality
    subset specifically)
  * every one of the 16 hand-curated manufacturing terms was correct
  * the generated layer showed 8.7% clearly-wrong mappings (dense agreement < 0.35)
    vs 0% for the curated manufacturing and workflow presets

The cause is structural, not a query bug. For Freigabe / Ausfall / Getriebe /
Relais / Drehmoment / Durchfluss Wikidata has NO German lexeme with a linked
manufacturing sense at all, and for Störung / Ausschuss / Dichtung the only linked
sense belongs to a different domain (ecology / committee / art). More querying
cannot invent absent data.

LICENCE
-------
This list is ORIGINAL WORK by the klix author and is licensed MIT, like the
package. It has to be original, because every candidate source that could have
supplied it automatically is copyleft or share-alike and therefore incompatible
with MIT — each checked at the source:

  dbnary / Wiktionary   CC BY-SA 4.0     (Wikimedia rightsinfo API)
  FreeDict deu-eng      GPLv2+ / AGPLv3  (COPYING + TEI header of the release)
  Apertium eng-deu      GPL-3.0          (GitHub API)
  PanLex                unverifiable     (endpoint unreachable from this host)
  Wikidata              CC0 1.0          <- the only compatible source

CONVENTIONS
-----------
  * one term maps to exactly ONE concept across ALL THREE domains. klix's index is
    flat (`word -> concept`), so a term appearing twice is an ambiguous mapping —
    `Glossary.validate()` flags it and the build fails. This constraint drove
    several term reassignments (documented inline where non-obvious).
  * ASCII and umlaut spellings are both listed where the umlaut form is common in
    typed tickets ("foerderband" and "förderband").
  * no proper names, brands, places, taxonomy, or function words.

Format matches klix.glossary: {CONCEPT: {"de": [...], "en": [...]}}
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. MANUFACTURING / PRODUCTION (shop floor, maintenance, quality)
# ---------------------------------------------------------------------------
MANUFACTURING: dict[str, dict[str, list[str]]] = {
    "conveyor": {"de": ["foerderband", "förderband", "transportband"],
                 "en": ["conveyor", "conveyor belt"]},
    "cycle_time": {"de": ["taktzeit", "zykluszeit"],
                   "en": ["cycle time", "cycle-time"]},
    "downtime": {"de": ["stillstand", "ausfallzeit", "anlagenstillstand"],
                 "en": ["downtime", "line stoppage", "standstill"]},
    "maintenance": {"de": ["wartung", "instandhaltung", "reparatur"],
                    "en": ["maintenance", "repair"]},
    "spare_part": {"de": ["ersatzteil", "ersatzteile"],
                   "en": ["spare part", "spare parts"]},
    "shift": {"de": ["schicht", "schichtbetrieb", "fruehschicht"],
              "en": ["shift", "work shift"]},
    "hydraulic": {"de": ["hydraulik", "hydraulikoel"],
                  "en": ["hydraulic", "hydraulic system"]},
    "pneumatic": {"de": ["pneumatik", "druckluft"],
                  "en": ["pneumatic", "compressed air"]},
    "sensor": {"de": ["sensor", "sensorik", "messfuehler"],
               "en": ["sensor", "sensing"]},
    "calibration": {"de": ["kalibrierung", "justierung", "abgleich"],
                    "en": ["calibration", "adjustment"]},
    "scrap": {"de": ["ausschuss", "fehlteil", "ausschussquote"],
              "en": ["scrap", "reject part"]},
    "warehouse": {"de": ["lager", "lagerbestand", "materiallager"],
                  "en": ["warehouse", "inventory", "stock"]},
    "safety_guard": {"de": ["schutzgitter", "schutztuer", "schutzabdeckung"],
                     "en": ["safety guard", "safety door", "guard"]},
    "error_code": {"de": ["fehlercode", "stoerungscode"],
                   "en": ["error code", "fault code"]},
    "commissioning": {"de": ["inbetriebnahme", "inbetriebsetzung"],
                      "en": ["commissioning", "startup"]},
    "batch": {"de": ["charge", "losgroesse"],
              "en": ["batch", "lot size"]},
    "assembly": {"de": ["montage", "montagelinie", "zusammenbau"],
                 "en": ["assembly", "assembly line"]},
    "gear": {"de": ["getriebe", "zahnrad", "uebersetzung"],
             "en": ["gearbox", "gear", "transmission"]},
    "torque": {"de": ["drehmoment", "anzugsmoment"],
               "en": ["torque", "tightening torque"]},
    "seal": {"de": ["dichtung", "dichtungsring", "abdichtung"],
             "en": ["seal", "gasket", "sealing"]},
    "wear": {"de": ["verschleiss", "abnutzung"],
             "en": ["wear", "abrasion"]},
    "failure": {"de": ["ausfall", "stoerung", "fehlfunktion"],
                "en": ["failure", "fault", "malfunction"]},
    "lubrication": {"de": ["schmierung", "schmierstoff"],
                    "en": ["lubrication", "lubricant"]},
    "tolerance": {"de": ["toleranz", "abmass"],
                  "en": ["tolerance"]},
    "flow_rate": {"de": ["durchfluss", "durchflussmenge", "volumenstrom"],
                  "en": ["flow rate", "flow"]},
    "pressure": {"de": ["druck", "luftdruck", "systemdruck"],
                 "en": ["pressure", "system pressure"]},
    "temperature": {"de": ["temperatur", "betriebstemperatur"],
                    "en": ["temperature", "operating temperature"]},
    "voltage": {"de": ["spannung", "versorgungsspannung"],
                "en": ["voltage", "supply voltage"]},
    "relay": {"de": ["relais", "schuetz"],
              "en": ["relay", "contactor"]},
    "valve": {"de": ["ventil", "absperrventil"],
              "en": ["valve", "shut-off valve"]},
    "pump": {"de": ["pumpe", "foerderpumpe"],
             "en": ["pump", "feed pump"]},
    "motor": {"de": ["motor", "elektromotor"],
              "en": ["motor", "electric motor"]},
    "switch": {"de": ["schalter", "taster", "endschalter"],
               "en": ["switch", "push button", "limit switch"]},
    "fuse": {"de": ["sicherung", "absicherung"],
             "en": ["fuse", "circuit protection"]},
    "cable": {"de": ["kabel", "leitung", "steckverbinder"],
              "en": ["cable", "wire", "connector"]},
    "screw": {"de": ["schraube", "gewinde", "schraubverbindung"],
              "en": ["screw", "thread", "bolted joint"]},
    "nut": {"de": ["mutter", "sechskantmutter"],
            "en": ["nut", "hex nut"]},
    "inspection": {"de": ["pruefung", "inspektion", "sichtpruefung"],
                   "en": ["inspection", "check", "examination"]},
    "approval": {"de": ["freigabe", "abnahme"],
                 "en": ["approval", "acceptance"]},
    "quality": {"de": ["qualitaet", "guete", "beschaffenheit"],
                "en": ["quality", "grade"]},
    "measurement": {"de": ["messung", "messwert", "messergebnis"],
                    "en": ["measurement", "measured value", "reading"]},
    "process": {"de": ["prozess", "verfahren", "ablauf"],
                "en": ["process", "procedure", "workflow"]},
    "output": {"de": ["ausgangsleistung", "durchsatz"],
               "en": ["output", "power output"]},
    "failure_cause": {"de": ["stoergrund", "ausfallgrund"],
                      "en": ["failure cause", "downtime reason"]},
}

# ---------------------------------------------------------------------------
# 2. IT / SOFTWARE / INFRASTRUCTURE
# ---------------------------------------------------------------------------
IT: dict[str, dict[str, list[str]]] = {
    "server": {"de": ["server", "host", "rechensystem"],
               "en": ["server", "host"]},
    "client": {"de": ["endgeraet", "arbeitsplatzrechner"],
               "en": ["client", "endpoint"]},
    "network": {"de": ["netzwerk", "netz", "verbindung"],
                "en": ["network", "connection", "link"]},
    "database": {"de": ["datenbank", "datenspeicher"],
                 "en": ["database", "datastore"]},
    "backup": {"de": ["sicherungskopie", "datensicherung"],
               "en": ["backup", "data backup"]},
    "restore": {"de": ["wiederherstellung", "ruecksicherung"],
                "en": ["restore", "recovery"]},
    "password": {"de": ["passwort", "kennwort", "zugangscode"],
                 "en": ["password", "credential", "passphrase"]},
    "login": {"de": ["anmeldung", "einloggen", "anmeldedaten"],
              "en": ["login", "log in", "sign in"]},
    "access_rights": {"de": ["zugriffsrechte", "berechtigung"],
                      "en": ["access rights", "permission"]},
    "error_message": {"de": ["fehlermeldung", "fehlertext"],
                      "en": ["error message", "error text"]},
    "bug": {"de": ["programmfehler", "softwarefehler", "absturz"],
            "en": ["bug", "defect", "crash"]},
    "issue_ticket": {"de": ["ticket", "vorgang", "stoerungsmeldung"],
                     "en": ["ticket", "incident"]},
    "deployment": {"de": ["bereitstellung", "ausrollen"],
                   "en": ["deployment", "rollout"]},
    "update": {"de": ["aktualisierung", "update", "patch"],
               "en": ["update", "patch", "upgrade"]},
    "interface": {"de": ["schnittstelle", "anbindung", "api"],
                  "en": ["interface", "integration", "api"]},
    "configuration": {"de": ["konfiguration", "einstellung", "parameter"],
                      "en": ["configuration", "setting", "parameter"]},
    "monitoring": {"de": ["ueberwachung", "monitoring"],
                   "en": ["monitoring", "supervision"]},
    "storage": {"de": ["speicher", "festplatte", "speicherplatz"],
                "en": ["storage", "disk", "disk space"]},
    "memory": {"de": ["arbeitsspeicher", "hauptspeicher"],
               "en": ["memory", "ram"]},
    "cpu": {"de": ["prozessor", "rechnerleistung"],
            "en": ["processor", "cpu"]},
    "application": {"de": ["anwendung", "applikation", "programm"],
                    "en": ["application", "app", "program"]},
    "service": {"de": ["dienst", "webdienst"],
                "en": ["service", "daemon", "web service"]},
    "firewall": {"de": ["firewall", "paketfilter"],
                 "en": ["firewall", "packet filter"]},
    "encryption": {"de": ["verschluesselung", "kryptografie"],
                   "en": ["encryption", "cryptography"]},
    # 'certificate' lives here as the TLS sense; the HR sense is 'zeugnis'
    # (everyday/certificate_hr) so the English word is not duplicated.
    "certificate": {"de": ["zertifikat", "tlszertifikat"],
                    "en": ["tls certificate", "digital certificate"]},
    "security_incident": {"de": ["sicherheitsvorfall", "angriff", "datenschutzvorfall"],
                          "en": ["security incident", "attack", "breach"]},
    "malware": {"de": ["schadsoftware", "virus", "trojaner"],
                "en": ["malware", "virus", "trojan"]},
    "firewall_rule": {"de": ["regel", "portfreigabe"],
                      "en": ["rule", "port release"]},
    "log_file": {"de": ["logdatei", "protokolldatei"],
                 "en": ["log file", "log"]},
    "query": {"de": ["abfrage", "suchabfrage"],
              "en": ["query"]},
    "timeout": {"de": ["zeitueberschreitung", "antwortzeit"],
                "en": ["timeout", "latency", "response time"]},
    "outage": {"de": ["unerreichbarkeit", "wartungsfenster"],
               "en": ["outage", "maintenance window"]},
    "bug_report": {"de": ["fehlerbericht", "fehlerbeschreibung"],
                   "en": ["bug report", "defect description"]},
    "version": {"de": ["version", "versionsstand"],
                "en": ["version", "build", "release"]},
    "rollback": {"de": ["zuruecksetzen", "rueckabwicklung"],
                 "en": ["rollback", "revert"]},
    "maintenance_plan": {"de": ["wartungsplan", "wartungsplanung"],
                         "en": ["maintenance plan", "maintenance schedule"]},
    "documentation": {"de": ["dokumentation", "anleitung", "handbuch"],
                      "en": ["documentation", "manual", "guide"]},
    "user_account": {"de": ["benutzerkonto", "benutzer"],
                     "en": ["user account", "user"]},
    "expiry": {"de": ["gueltigkeit", "ablaufdatum"],
               "en": ["expiry", "validity", "expiration date"]},
    # 'leistung' is the IT/performance sense here; the power sense is
    # manufacturing/output ('ausgangsleistung'). One word, one concept.
    "performance": {"de": ["leistung", "geschwindigkeit"],
                    "en": ["performance", "speed", "throughput"]},
}

# ---------------------------------------------------------------------------
# 3. EVERYDAY / OFFICE (the hardest layer: heavy polysemy)
# ---------------------------------------------------------------------------
EVERYDAY: dict[str, dict[str, list[str]]] = {
    "invoice": {"de": ["rechnung", "abrechnung"],
                "en": ["invoice", "bill", "statement"]},
    "refund": {"de": ["erstattung", "rueckerstattung", "gutschrift"],
               "en": ["refund", "reimbursement", "credit note"]},
    "delivery": {"de": ["lieferung", "zustellung", "versand"],
                 "en": ["delivery", "shipment", "dispatch"]},
    "order": {"de": ["bestellung", "auftrag"],
              "en": ["order", "purchase order"]},
    "cancellation": {"de": ["stornierung", "kuendigung", "abbruch"],
                     "en": ["cancellation", "termination", "abort"]},
    "appointment": {"de": ["termin", "besprechung"],
                    "en": ["appointment", "meeting"]},
    "deadline": {"de": ["frist", "stichtag"],
                 "en": ["deadline", "due date"]},
    "contract": {"de": ["vertrag", "vereinbarung"],
                 "en": ["contract", "agreement"]},
    "department": {"de": ["abteilung", "bereich", "ressort"],
                   "en": ["department", "division", "unit"]},
    "employee": {"de": ["mitarbeiter", "beschaeftigter", "angestellter"],
                 "en": ["employee", "staff member", "worker"]},
    "supervisor": {"de": ["vorgesetzter", "leiter", "fuehrungskraft"],
                   "en": ["supervisor", "manager", "lead"]},
    "vacation": {"de": ["urlaub", "erholungsurlaub", "freizeitausgleich"],
                 "en": ["vacation", "annual leave", "time off"]},
    "sick_leave": {"de": ["krankmeldung", "krankschreibung", "arbeitsunfaehigkeit"],
                   "en": ["sick leave", "sick note", "absence"]},
    "salary": {"de": ["gehalt", "lohn", "verguetung"],
               "en": ["salary", "wage", "pay"]},
    "payment": {"de": ["zahlung", "ueberweisung"],
                "en": ["payment", "transfer"]},
    # 'account'/'konto' are the bank sense here; the IT sense is 'benutzerkonto'.
    "bank_account": {"de": ["konto", "bankkonto"],
                     "en": ["account", "bank account"]},
    "receipt": {"de": ["quittung", "kassenzettel"],
                "en": ["receipt", "proof of purchase"]},
    "warranty": {"de": ["garantie", "gewaehrleistung"],
                 "en": ["warranty", "guarantee"]},
    "complaint": {"de": ["beschwerde", "reklamation", "beanstandung"],
                  "en": ["complaint", "reclamation"]},
    "return_goods": {"de": ["rueckgabe", "retoure", "umtausch"],
                     "en": ["return", "exchange"]},
    "supplier": {"de": ["lieferant", "zulieferer", "anbieter"],
                 "en": ["supplier", "vendor", "provider"]},
    "customer": {"de": ["kunde", "auftraggeber"],
                 "en": ["customer", "buyer"]},
    "approval_office": {"de": ["genehmigung", "zulassung", "erlaubnis"],
                        "en": ["permit", "license", "authorization"]},
    "request": {"de": ["anfrage", "anforderung"],
                "en": ["request", "requirement"]},
    "change_request": {"de": ["aenderung", "aenderungsantrag", "anpassung"],
                       "en": ["change", "change request", "modification"]},
    "training": {"de": ["schulung", "weiterbildung", "unterweisung"],
                 "en": ["training", "course", "instruction"]},
    "onboarding": {"de": ["einarbeitung", "einstieg", "einarbeitungsplan"],
                   "en": ["onboarding", "induction"]},
    # HR sense of 'certificate'; the TLS sense is IT/certificate ('zertifikat').
    "certificate_hr": {"de": ["zeugnis", "bescheinigung", "nachweis"],
                       "en": ["reference letter", "proof of qualification"]},
    "job_application": {"de": ["bewerbung", "bewerbungsunterlagen", "lebenslauf"],
                        "en": ["job application", "cv", "resume"]},
    "notice_period": {"de": ["kuendigungsfrist", "austrittsdatum"],
                      "en": ["notice period", "termination date"]},
    "overtime": {"de": ["ueberstunden", "mehrarbeit"],
                 "en": ["overtime", "extra hours"]},
    "expense": {"de": ["ausgabe", "spesen", "kosten"],
                "en": ["expense", "cost", "spending"]},
    "budget": {"de": ["budget", "etat", "kostenrahmen"],
               "en": ["budget", "funds"]},
    "report": {"de": ["bericht", "auswertung", "meldung"],
               "en": ["report", "analysis"]},
    "meeting_notes": {"de": ["protokoll", "sitzungsprotokoll", "notizen"],
                      "en": ["minutes", "meeting notes"]},
    "question": {"de": ["frage", "rueckfrage", "nachfrage"],
                 "en": ["question", "clarification"]},
    "urgent": {"de": ["dringend", "eilig", "sofort", "vorrangig"],
               "en": ["urgent", "asap", "immediately", "priority"]},
    "problem": {"de": ["problem", "schwierigkeit"],
                "en": ["problem", "issue"]},
    "delay": {"de": ["verzoegerung", "verspaetung", "verzug"],
              "en": ["delay", "lag"]},
    "confirmation": {"de": ["bestaetigung", "zusage"],
                     "en": ["confirmation", "acknowledgement"]},
}


DOMAINS = {"manufacturing": MANUFACTURING, "it": IT, "everyday": EVERYDAY}


def all_domains() -> dict[str, dict[str, dict[str, list[str]]]]:
    """{domain: mapping} — the three curated domains, kept separate."""
    return DOMAINS


def merge_sources(*sources, strict: bool = True) -> dict[str, dict[str, list[str]]]:
    """Merges ANY number of named term sources into one concept map.

    This is the generic form of the old `merged()`: it takes
    ``(name, mapping)`` pairs rather than reading a module-level `DOMAINS`
    constant, so a fourth, fifth, or externally supplied glossary goes through
    exactly the same path as the three built-ins. There is no notion of a
    "domain" here — `name` is only used to disambiguate a concept key and to
    label the error message. Domain-ness lives in the `tags` metadata.

    Parameters
    ----------
    *sources:
        ``(name, mapping)`` pairs, in priority order.
    strict:
        When True (default) a term appearing under two concepts raises
        `ValueError`. This is the same single conflict rule the engine applies
        (`Glossary.conflicts`); keeping it identical here means a build cannot
        ship something the engine would refuse to merge later.

    Language buckets are created on demand — a concept with only `{"fr": [...]}`
    is carried through as-is instead of gaining empty `de`/`en` lists.
    """
    out: dict[str, dict[str, list[str]]] = {}
    owner: dict[str, tuple[str, str]] = {}
    for name, mapping in sources:
        for concept, langs in mapping.items():
            if not isinstance(langs, dict):
                raise ValueError(f"{name}/{concept}: expected a language map")
            key = concept if concept not in out else f"{name}_{concept}"
            bucket = out.setdefault(key, {})
            for lang, terms in langs.items():
                if lang == "tags":            # metadata, not a language
                    continue
                own = bucket.setdefault(lang, [])
                for term in terms:
                    low = term.lower()
                    prev = owner.get(low)
                    if prev is not None and prev != (name, concept):
                        raise ValueError(
                            f"term {term!r} appears in {name}/{concept} and already "
                            f"belongs to {prev[0]}/{prev[1]} — the klix index is flat "
                            f"(one word -> one concept), so this is an ambiguous "
                            f"mapping, not a merge detail"
                        )
                    owner[low] = (name, concept)
                    if low not in own:
                        own.append(low)
    return out


def merged(*sources, strict: bool = True) -> dict[str, dict[str, list[str]]]:
    """All curated domains as ONE mapping (klix uses a single flat index).

    Kept as the zero-argument entry point for the built-in set; it delegates to
    `merge_sources`, so adding a domain is adding an entry to `DOMAINS` (or
    calling `merge_sources` yourself) and nothing else.

    Raises on a cross-source term collision rather than silently overriding:
    a flat index can only resolve a word to one concept, so a duplicate is an
    ambiguous mapping. No concept is ever renamed — the name-prefixed key is only
    used for a genuinely NEW concept that happens to share a key.
    """
    return merge_sources(*DOMAINS.items(), strict=strict)


def tags() -> dict[str, list[str]]:
    """concept -> [domain tag]. Free metadata; carries no validation meaning."""
    return {concept: [domain] for domain, mapping in DOMAINS.items()
            for concept in mapping}


def counts() -> dict[str, int]:
    return {d: len(m) for d, m in DOMAINS.items()}


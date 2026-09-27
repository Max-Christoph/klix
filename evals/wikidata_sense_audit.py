"""Is the glossary problem a BUG or a DATA-BASIS problem? (corrected revision)

Two bugs in the first attempt are fixed here, both of which produced false
"NO ITEMS AT ALL" verdicts:

  1. LEMMA SPELLING — German lexemes are stored with their real orthography
     ("Prüfung", "Störung", "Qualität"). Querying the ASCII-folded form
     ("pruefung") finds nothing, which is a query artefact, not a data gap.
  2. LANGUAGE OF THE TECH TEST — itemLabel/itemDescription come back in GERMAN
     by default, so an English keyword regex flagged almost nothing. The test is
     now German (plus the English label as a fallback).

For each probe term this enumerates every German lexeme with that lemma, all of
its senses, and the item each sense points at via P5137 — then asks whether ANY
of those items sits in the manufacturing/technical domain.

Run: uv run python -m evals.wikidata_sense_audit
"""
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ENDPOINT = "https://query.wikidata.org/sparql"
UA = "klix-sense-audit/0.2 (https://github.com/Max-Christoph/klix)"

# German lemma (REAL orthography) -> the sense a manufacturing glossary needs
PROBE = [
    ("Mutter", "Gewinde-Mutter (fastener)"),
    ("Messung", "Messung (measurement)"),
    ("Prüfung", "Prüfung (test/inspection)"),
    ("Freigabe", "Freigabe (release/approval)"),
    ("Leistung", "Leistung (power/output)"),
    ("Störung", "Störung (fault)"),
    ("Prozess", "Prozess (process)"),
    ("Qualität", "Qualität (quality)"),
    ("Verschleiß", "Verschleiß (wear)"),
    ("Dichtung", "Dichtung (seal/gasket)"),
    ("Ausfall", "Ausfall (failure)"),
    ("Getriebe", "Getriebe (gearbox)"),
    ("Relais", "Relais (relay)"),
    ("Drehmoment", "Drehmoment (torque)"),
    ("Durchfluss", "Durchfluss (flow rate)"),
    ("Temperatur", "Temperatur (temperature)"),
    ("Druck", "Druck (pressure)"),
    ("Spannung", "Spannung (voltage/stress)"),
    ("Schmierung", "Schmierung (lubrication)"),
    ("Toleranz", "Toleranz (tolerance)"),
    ("Wartung", "Wartung (maintenance)"),
    ("Reparatur", "Reparatur (repair)"),
    ("Ersatzteil", "Ersatzteil (spare part)"),
    ("Lager", "Lager (warehouse/bearing)"),
    ("Schraube", "Schraube (screw)"),
    ("Ventil", "Ventil (valve)"),
    ("Pumpe", "Pumpe (pump)"),
    ("Motor", "Motor (motor)"),
    ("Schalter", "Schalter (switch)"),
    ("Kabel", "Kabel (cable)"),
    ("Sicherung", "Sicherung (fuse)"),
    ("Förderband", "Förderband (conveyor)"),
    ("Taktzeit", "Taktzeit (cycle time)"),
    ("Stillstand", "Stillstand (downtime)"),
    ("Kalibrierung", "Kalibrierung (calibration)"),
    ("Ausschuss", "Ausschuss (scrap)"),
    ("Fehler", "Fehler (error)"),
    ("Instandhaltung", "Instandhaltung (maintenance)"),
    ("Montage", "Montage (assembly)"),
    ("Fertigung", "Fertigung (manufacturing)"),
]

# GERMAN technical-domain marker (item labels/descriptions arrive in German).
TECH_DE = re.compile(
    r"(mutterschraube|schraube|gewinde|befestigung|messger|messung|messen|messwert|"
    r"leistung|ausgangsleistung|leistungsabgabe|prüf|test|inspektion|kontrolle|"
    r"freigabe|genehmigung|zulassung|störung|fehler|defekt|mangel|havarie|"
    r"prozess|verfahren|ablauf|qualität|güte|beschaffenheit|"
    r"verschleiß|abnutzung|abdichtung|dichtung|dichtungsring|ausfall|"
    r"getriebe|übersetzung|relais|drehmoment|moment|"
    r"durchfluss|durchflussmenge|strömung|temperatur|druck|spannung|"
    r"schmierung|schmierstoff|toleranz|abmaß|"
    r"wartung|instandhaltung|reparatur|ersatzteil|"
    r"lager|speicher|schraube|ventil|pumpe|motor|schalter|kabel|sicherung|"
    r"bauteil|baugruppe|maschine|maschinenelement|gerät|vorrichtung|werkzeug|"
    r"anlage|fabrik|werk|produktion|fertigung|montage|"
    r"hydraulik|pneumatik|sensor|schaltung|antrieb|steuerung|regelung)",
    re.IGNORECASE,
)
TECH_EN = re.compile(
    r"\b(nut|screw|bolt|fastener|thread|measur\w*|power|output|test\w*|inspect\w*|"
    r"release|approv\w*|fault|malfunction|error|failure|defect|process|quality|"
    r"wear|seal|gasket|gearbox|relay|torque|flow|pressure|voltage|stress|"
    r"lubric\w*|tolerance|maintenance|repair|spare part|warehouse|bearing|"
    r"valve|pump|motor|switch|cable|fuse|conveyor|cycle time|downtime|"
    r"calibrat\w*|scrap|assembly|manufactur\w*|machine|component|tool|circuit|"
    r"hydraulic|pneumatic|sensor|electric\w*|mechanic\w*|industrial|plant)\b",
    re.IGNORECASE,
)

CACHE = {}


def sparql(query, timeout=120, retries=4, polite=1.2):
    if query in CACHE:
        return CACHE[query]
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query})
    req = urllib.request.Request(url, headers={
        "Accept": "application/sparql-results+json", "User-Agent": UA})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                rows = json.loads(r.read().decode("utf-8"))["results"]["bindings"]
            CACHE[query] = rows
            time.sleep(polite)
            return rows
        except (urllib.error.URLError, TimeoutError, KeyError) as exc:
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"query failed after {retries}: {last}")


# Two queries: one for the lexeme's senses, one for the item labels. Joining
# senses+labels in one query repeatedly timed out on WDQS.
# CRITICAL: `wikibase:lemma "Mutter"` matches NOTHING — lexeme lemmas are
# language-tagged literals ("Mutter"@de) and an untagged literal does not compare
# equal to them in SPARQL. That silently produced 0 senses for all 40 terms in
# the first revision of this audit. Filter on STR() instead.
SENSES_Q = """
SELECT DISTINCT ?sense ?item WHERE {
  ?lexeme dct:language wd:Q188 ; wikibase:lemma ?l ;
          wikibase:lexicalCategory ?pos ; ontolex:sense ?sense .
  FILTER(STR(?l) = "%s")
  OPTIONAL { ?sense wdt:P5137 ?item . }
}
LIMIT 80
"""

LABELS_Q = """
SELECT ?item ?itemLabel ?itemDescription WHERE {
  VALUES ?item { %s }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "de,en". }
}
LIMIT 80
"""


def audit(lemma):
    rows = sparql(SENSES_Q % lemma)
    senses = {r["sense"]["value"] for r in rows if "sense" in r}
    items = {r["item"]["value"].rsplit("/", 1)[-1] for r in rows if "item" in r}
    if not items:
        return len(senses), {}, {}
    vals = " ".join(f"wd:{q}" for q in items)
    labs = sparql(LABELS_Q % vals)
    info = {}
    for r in labs:
        qid = r["item"]["value"].rsplit("/", 1)[-1]
        lab = r.get("itemLabel", {}).get("value", "")
        desc = r.get("itemDescription", {}).get("value", "")
        info[qid] = (lab, desc)
    tech = {q: v for q, v in info.items()
            if TECH_DE.search(v[0] + " " + v[1]) or TECH_EN.search(v[0] + " " + v[1])}
    return len(senses), info, tech


def main():
    print("=" * 104)
    print("WIKIDATA SENSE AUDIT v2 — does a manufacturing sense EXIST for these terms?")
    print("=" * 104)
    print("  BUG  = a technical item is linked -> builder picked the wrong sense -> fixable")
    print("  DATA = no technical item linked at all -> not fixable by better querying\n")
    print(f"{'lemma':16} {'need':28} {'senses':>7} {'items':>6} {'tech':>5}  verdict")
    print("-" * 104)

    n_tech = n_data = 0
    fixables, gaps, failed = [], [], []
    for lemma, need in PROBE:
        try:
            n_s, info, tech = audit(lemma)
        except RuntimeError as exc:
            print(f"{lemma:16} {need:28} {'':>7} {'':>6} {'':>5}  QUERY FAILED")
            failed.append(lemma)
            continue
        if tech:
            verdict = f"BUG? technical item present: {list(tech.values())[0][0][:30]!r}"
            n_tech += 1
            fixables.append((lemma, [v[0] for v in tech.values()][:3]))
        else:
            verdict = "DATA GAP"
            n_data += 1
            gaps.append((lemma, [f"{v[0]}" for v in list(info.values())[:3]]))
        print(f"{lemma:16} {need:28} {n_s:>7} {len(info):>6} {len(tech):>5}  {verdict}")

    print()
    print(f"  technical sense EXISTS in Wikidata : {n_tech:2d}   <- fixable by disambiguation")
    print(f"  NO technical sense linked          : {n_data:2d}   <- data gap, not fixable")
    if failed:
        print(f"  query failed (inconclusive)        : {len(failed)}  {failed}")

    if fixables:
        print("\n  FIXABLE (technical item already linked, builder picked another sense):")
        for lemma, labs in fixables:
            print(f"    {lemma:16} -> {labs}")
    if gaps:
        print("\n  DATA GAPS (what IS linked instead):")
        for lemma, labs in gaps:
            print(f"    {lemma:16} -> {labs}")


if __name__ == "__main__":
    main()

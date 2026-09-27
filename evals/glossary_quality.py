"""Glossary quality audit: is the EN/DE coverage real, and is it useful?

Answers, with numbers (not adjectives):
  A. per-language distribution (terms per language, concepts covered each side)
  B. concept-size distribution, multi-word share, duplicate/ambiguity state
  C. manufacturing/process vocabulary coverage against a concrete probe list
  D. 20 random entries per language, printed verbatim
  E. provenance: which classes contributed (from build_meta.json)

Run: uv run python -m evals.glossary_quality
"""
import collections
import json
import random
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from klix.glossary import DEFAULT_GLOSSARY, Glossary  # noqa: E402


def load():
    raw = json.loads(DEFAULT_GLOSSARY.read_text(encoding="utf-8"))
    return raw


def section(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


# Probe vocabulary: what a manufacturing/process glossary MUST cover to be
# useful for the stated purpose (shop-floor tickets, maintenance, quality, IT).
PROBE = {
    "core_manufacturing": [
        ("foerderband", "conveyor"), ("taktzeit", "cycle time"),
        ("stillstand", "downtime"), ("wartung", "maintenance"),
        ("ersatzteil", "spare part"), ("schicht", "shift"),
        ("hydraulik", "hydraulic"), ("pneumatik", "pneumatic"),
        ("sensor", "sensor"), ("kalibrierung", "calibration"),
        ("ausschuss", "scrap"), ("lager", "warehouse"),
        ("schutzgitter", "safety guard"), ("fehlercode", "error code"),
        ("inbetriebnahme", "commissioning"), ("charge", "batch"),
    ],
    "process_quality": [
        ("prozess", "process"), ("qualitaet", "quality"),
        ("messung", "measurement"), ("pruefung", "inspection"),
        ("toleranz", "tolerance"), ("freigabe", "approval"),
        ("storung", "malfunction"), ("verschleiss", "wear"),
        ("drehmoment", "torque"), ("druck", "pressure"),
        ("temperatur", "temperature"), ("durchfluss", "flow rate"),
        ("spannung", "voltage"), ("leistung", "power"),
        ("schmierung", "lubrication"), ("dichtung", "seal"),
    ],
    "it_office": [
        ("rechner", "computer"), ("speicher", "memory"),
        ("netzwerk", "network"), ("benutzer", "user"),
        ("passwort", "password"), ("datei", "file"),
        ("rechnung", "invoice"), ("lieferung", "delivery"),
        ("vertrag", "contract"), ("termin", "appointment"),
        ("abteilung", "department"), ("mitarbeiter", "employee"),
    ],
}


def main():
    raw = load()
    g = Glossary(raw)
    concepts = list(raw)

    section("A. PER-LANGUAGE DISTRIBUTION")
    per_lang_terms = collections.Counter()
    per_lang_concepts = collections.Counter()
    for k, langs in raw.items():
        for lg, terms in langs.items():
            per_lang_terms[lg] += len(terms)
            if terms:
                per_lang_concepts[lg] += 1
    print(f"  concepts total            : {len(raw)}")
    for lg in sorted(per_lang_terms):
        print(f"  terms in {lg!r:6}          : {per_lang_terms[lg]:6d}")
        print(f"  concepts with {lg!r:6} terms: {per_lang_concepts[lg]:6d}")
    balanced = all(per_lang_concepts[lg] == len(raw) for lg in per_lang_concepts)
    print(f"  every concept carries BOTH languages: {balanced}")
    sizes = [len(v) for v in raw.values()]
    print(f"  avg terms/concept         : {sum(sizes)/len(sizes):.2f}")
    print(f"  concepts with 1 term/side : {sum(1 for v in raw.values() if len(v['de'])==1 and len(v['en'])==1)}")
    print(f"  concepts with >2 terms/side: {sum(1 for v in raw.values() if len(v['de'])>2 or len(v['en'])>2)}")

    section("B. STRUCTURE / AMBIGUITY")
    multi = sum(1 for v in raw.values() if any(" " in t for t in v["de"] + v["en"]))
    print(f"  concepts containing a multi-word term: {multi} ({multi/len(raw):.1%})")
    maxw = max(len(t.split()) for v in raw.values() for t in v["de"] + v["en"])
    print(f"  longest term (in tokens)  : {maxw}")
    findings = g.validate()
    print(f"  validate() findings       : {len(findings)}  {findings[:2]}")
    idx = g._index
    print(f"  unique lookup keys        : {len(idx)}")
    dup = len(idx) - len({v for v in idx.values()})
    print(f"  keys mapping to a concept : {len(set(idx.values()))}")

    section("C. DOMAIN COVERAGE (probe vocabulary)")
    for group, pairs in PROBE.items():
        hit_de = hit_en = 0
        missing = []
        for de, en in pairs:
            if de in idx:
                hit_de += 1
            if en in idx:
                hit_en += 1
            if de not in idx and en not in idx:
                missing.append(f"{de}/{en}")
        print(f"  {group:20s} DE {hit_de:2d}/{len(pairs)}   EN {hit_en:2d}/{len(pairs)}"
              f"   both-missing {len(missing)}: {missing}")

    section("D. 20 RANDOM ENTRIES PER LANGUAGE (verbatim)")
    rng = random.Random(20260927)
    for lg, other in (("de", "en"), ("en", "de")):
        print(f"\n  --- {lg.upper()} side (20 random concepts) ---")
        for k in rng.sample(concepts, 20):
            print(f"    {k:34s} {lg}={raw[k][lg]!r:58s} {other}={raw[k][other]!r}")

    section("E. PROVENANCE (build_meta.json)")
    meta_path = DEFAULT_GLOSSARY.with_name("build_meta.json")
    if meta_path.exists():
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"  source            : {m['source']}")
        print(f"  contributed clases: {len(m['classes_contributed'])}")
        print(f"  skipped classes   : {len(m['classes_skipped'])}")
        for s in m["classes_skipped"]:
            print(f"      - {s['class']} ({s['qid']}): {s['reason'][:52]}")
        print(f"  classes           : {sorted(m['classes_contributed'])}")
    else:
        print("  (no build_meta.json)")


if __name__ == "__main__":
    main()

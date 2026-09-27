"""Correctness audit: does each domain term map to the RIGHT concept?

The coverage probe in glossary_quality.py only asks "is the word present". A
present-but-wrong mapping is worse than a missing one: it silently bridges the
query to the wrong English terms. This script checks the MAPPING QUALITY on a
curated domain vocabulary and prints the concept + English side for eyeballing.

Run: uv run python -m evals.glossary_correctness
"""
import json
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from klix.glossary import DEFAULT_GLOSSARY, Glossary  # noqa: E402

# German term -> the English term it SHOULD bridge to (expected concept sense).
EXPECT = [
    ("qualitaet", "quality"), ("qualität", "quality"),
    ("prozess", "process"), ("messung", "measurement"),
    ("pruefung", "test"), ("prüfung", "test"),
    ("toleranz", "tolerance"), ("freigabe", "release"),
    ("verschleiss", "wear"), ("druck", "pressure"),
    ("temperatur", "temperature"), ("spannung", "voltage"),
    ("leistung", "power"), ("schmierung", "lubrication"),
    ("dichtung", "gasket"), ("fehler", "error"),
    ("stoerung", "fault"), ("störung", "fault"),
    ("ausfall", "failure"), ("wartung", "maintenance"),
    ("reparatur", "repair"), ("instandhaltung", "maintenance"),
    ("ersatzteil", "spare part"), ("lager", "warehouse"),
    ("foerderband", "conveyor"), ("förderband", "conveyor"),
    ("taktzeit", "cycle time"), ("stillstand", "downtime"),
    ("kalibrierung", "calibration"), ("ausschuss", "scrap"),
    ("schraube", "screw"), ("mutter", "nut"),
    ("ventil", "valve"), ("pumpe", "pump"),
    ("motor", "engine"), ("getriebe", "gearbox"),
    ("schalter", "switch"), ("kabel", "cable"),
    ("sicherung", "fuse"), ("relais", "relay"),
]


def main():
    raw = json.loads(DEFAULT_GLOSSARY.read_text(encoding="utf-8"))
    g = Glossary(raw)
    idx = g._index

    print("=" * 100)
    print("MAPPING CORRECTNESS — curated domain vocabulary")
    print("=" * 100)
    print(f"{'term':16} {'concept':22} {'english side':38} verdict")
    print("-" * 100)

    ok = wrong = missing = 0
    wrongs = []
    for term, expected in EXPECT:
        concept = idx.get(term)
        if concept is None:
            print(f"{term:16} {'—':22} {'—':38} MISSING")
            missing += 1
            continue
        en_side = raw[concept].get("en", [])
        en_str = ", ".join(en_side)[:36]
        # correctness: does the expected sense appear anywhere on the English side?
        if any(expected in t for t in en_side) or expected == concept:
            verdict = "ok"
            ok += 1
        else:
            verdict = f"WRONG (expected ~{expected!r})"
            wrong += 1
            wrongs.append((term, concept, en_str, expected))
        print(f"{term:16} {concept:22} {en_str:38} {verdict}")

    total = ok + wrong + missing
    print()
    print(f"  checked {total}: ok {ok}  WRONG {wrong}  missing {missing}")
    if wrong:
        print(f"  wrong-mapping rate: {wrong/total:.1%}")
    print()
    print("  WRONG MAPPINGS IN DETAIL (the harmful case: present but misleading):")
    for term, concept, en_str, expected in wrongs:
        print(f"    {term!r:18} -> concept {concept!r:22} en=[{en_str}]  expected ~{expected!r}")


if __name__ == "__main__":
    main()

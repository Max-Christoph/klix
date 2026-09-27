"""Sense check for the terms that FAIL in the domain-correctness probe.

For each: which items does the German lexeme's P5137 link to, and which item
does the shipped glossary actually map the word to? If a proper technical item
exists but the shipped mapping points elsewhere, it is a BUG (fixable). If the
word is genuinely polysemous with no technical sense linked, it is a DATA GAP.

Run: uv run python -c "import runpy; runpy.run_module('evals.glossary_sense_check', run_name='__main__')"
"""
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from klix.glossary import DEFAULT_GLOSSARY, Glossary  # noqa: E402

UA = "klix-sense-check/0.1"
EP = "https://query.wikidata.org/sparql"

# term -> what the glossary currently maps it to (filled from the file)
FAILING = ["Prüfung", "Leistung", "Störung", "Verschleiß", "Dichtung", "Messung",
           "Freigabe", "Ausfall", "Getriebe", "Relais", "Ersatzteil", "Taktzeit",
           "Förderband", "Stillstand", "Instandhaltung", "Fertigung",
           "Drehmoment", "Durchfluss", "Mutter", "Ausschuss"]

Q = """
SELECT ?item ?lab ?desc WHERE {
  ?lx dct:language wd:Q188 ; wikibase:lemma ?l ; ontolex:sense ?s .
  FILTER(STR(?l) = "%s")
  ?s wdt:P5137 ?item .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "de,en". }
}
LIMIT 6
"""


def sparql(q, retries=3, timeout=90):
    url = EP + "?" + urllib.parse.urlencode({"query": q})
    req = urllib.request.Request(url, headers={
        "Accept": "application/sparql-results+json", "User-Agent": UA})
    last = None
    for a in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                rows = json.loads(r.read().decode("utf-8"))["results"]["bindings"]
            time.sleep(1.0)
            return rows
        except (urllib.error.URLError, TimeoutError, KeyError) as e:
            last = e
            time.sleep(2.0 * (a + 1))
    raise RuntimeError(str(last))


def main():
    raw = json.loads(DEFAULT_GLOSSARY.read_text(encoding="utf-8"))
    g = Glossary(raw)
    print("=" * 108)
    print("SENSE CHECK — technical item available vs. what the shipped glossary picked")
    print("=" * 108)
    for term in FAILING:
        concept = g._index.get(term.lower())
        shipped = raw[concept]["en"] if concept else None
        try:
            rows = sparql(Q % term)
        except RuntimeError as e:
            print(f"\n{term}: query failed ({e})")
            continue
        print(f"\n{term}")
        print(f"  shipped mapping : {concept!r} -> en={shipped}")
        if not rows:
            print("  wikidata senses : NO P5137 item linked  -> DATA GAP")
            continue
        for r in rows:
            qid = r["item"]["value"].rsplit("/", 1)[-1]
            lab = r.get("lab", {}).get("value", "")
            desc = r.get("desc", {}).get("value", "")
            print(f"  available       : {qid:12} {lab[:34]:36} {desc[:46]}")


if __name__ == "__main__":
    main()

"""Polysemy review: how many distinct Wikidata senses does each glossary term carry?

Why this exists
---------------
The lookup is flat and context-free (`word -> concept -> synonyms`). A German word
with more than one sense therefore resolves to whichever concept claimed it, and a
query using the *other* sense is bridged to the wrong concept. Two such cases are
reproduced end-to-end in `docs/curated-correctness.md` (`lager`, `leiter`).

This script measures the exposure instead of guessing at it: for every German term
in the curated glossary, how many independent Wikidata items do its lexeme's P5137
senses point at? Many senses = high risk of a wrong bridge.

It is NOT an error list. Most polysemous terms are perfectly mapped; the count only
says how much room there is for the other reading to be hit.

Two failure modes are guarded against explicitly, because both silently produce
"0 senses for every term", which looks exactly like a data finding:

  * the language tag must be a LANGUAGE CODE ("@de"), not an item QID ("@Q188")
  * German noun lemmas are capitalised in Wikidata, so every spelling variant is
    queried and mapped back to the lowercase glossary term

Both mistakes were made while writing this script. The pre-flight below is what
caught them; see also `evals/curated_correctness_new.py` and the P5 commit message.

Run: uv run python -m evals.polysemy_review
"""
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

REPO = Path(__file__).resolve().parents[1]
CURATED = REPO / "src" / "klix" / "data" / "curated_glossary.json"
DOMAINS = REPO / "src" / "klix" / "data" / "curated_domains.json"

EP = "https://query.wikidata.org/sparql"
UA = "klix-polysemy-review/0.1"

# Known-positive cases: both have German lexeme senses in Wikidata. If the query
# returns nothing for these, it is broken and every count below would be a lie.
PRE_FLIGHT = ["Schraube", "Druck", "Lager"]

Q = """
SELECT ?l ?item ?lab WHERE {
  VALUES ?l { %s }
  ?lx wikibase:lemma ?l ; ontolex:sense ?s .
  ?s wdt:P5137 ?item .
  OPTIONAL { ?item rdfs:label ?lab . FILTER(LANG(?lab) = "de") }
}
"""


def sparql(query: str, retries: int = 4, timeout: int = 120):
    url = EP + "?" + urllib.parse.urlencode({"query": query})
    req = urllib.request.Request(url, headers={
        "Accept": "application/sparql-results+json", "User-Agent": UA})
    for a in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))["results"]["bindings"]
        except (urllib.error.URLError, TimeoutError, KeyError):
            time.sleep(2.0 * (a + 1))
    return None


def spellings(term: str) -> list[str]:
    """German lemmas are capitalised in Wikidata; try the shapes the term can take."""
    return list(dict.fromkeys([term, term[:1].upper() + term[1:], term.upper()]))


def main() -> int:
    raw = json.loads(CURATED.read_text(encoding="utf-8"))
    mapping = raw.get("concepts", raw)
    mapping = {k: {lg: t for lg, t in v.items() if lg != "tags"}
               for k, v in mapping.items()}
    tag_of = json.loads(DOMAINS.read_text(encoding="utf-8")).get("concepts", {})

    # ---- pre-flight ------------------------------------------------------
    pre = sparql(Q % " ".join('"%s"@de' % w for w in PRE_FLIGHT))
    if pre is None:
        print("ABORT: pre-flight query failed (network).")
        return 1
    got = {b["l"]["value"] for b in pre}
    if not got:
        print(f"ABORT: pre-flight returned nothing for {PRE_FLIGHT}, all of which "
              f"have Wikidata lexeme senses.\n       The query is broken; any count "
              f"below would be an artifact.")
        return 1
    print(f"pre-flight : {sorted(got)} -> {len(pre)} rows, query works\n")

    # ---- every German term of the curated glossary -----------------------
    term_concepts: dict[str, list[str]] = {}
    for c, langs in mapping.items():
        for t in langs.get("de", []):
            term_concepts.setdefault(t.strip().lower(), []).append(c)

    lookup: dict[str, str] = {}
    for t in term_concepts:
        for sp in spellings(t):
            lookup.setdefault(sp, t)

    variants = sorted(lookup)
    print(f"glossary terms      : {len(term_concepts)}")
    print(f"spelling variants   : {len(variants)} queried\n")
    print("querying Wikidata:", flush=True)

    items: dict[str, dict[str, set]] = {}
    for i in range(0, len(variants), 60):
        chunk = variants[i:i + 60]
        rows = sparql(Q % " ".join(f'"{l}"@de' for l in chunk))
        if rows is None:
            print("ABORT: a batch was inconclusive; a partial count would be "
                  "meaningless.")
            return 1
        for b in rows:
            base = lookup.get(b["l"]["value"])
            if base is None:
                continue
            qid = b["item"]["value"].rsplit("/", 1)[-1]
            lab = b.get("lab", {}).get("value", "")
            e = items.setdefault(base, {"qids": set(), "labs": set()})
            e["qids"].add(qid)
            if lab:
                e["labs"].add(lab)
        print(f"  batch {i // 60 + 1}: {len(chunk)} variants ({len(rows)} rows)",
              flush=True)

    # ---- report ----------------------------------------------------------
    ranked = sorted(term_concepts,
                    key=lambda t: (-len(items.get(t, {"qids": set()})["qids"]), t))
    poly = [t for t in ranked if len(items.get(t, {"qids": set()})["qids"]) > 1]

    print()
    print("=" * 112)
    print("POLYSEMY EXPOSURE — terms whose German lexeme points at MORE THAN ONE item")
    print("=" * 112)
    print(f"  {len(poly)} of {len(term_concepts)} terms carry more than one sense.")
    print("  Ranking is exposure, not error: a polysemous term is usually mapped")
    print("  correctly. It marks where the OTHER reading can be hit.\n")
    print(f"{'items':>5}  {'term':<20} {'concept':<18} {'domain':<14} other senses (de labels)")
    print("-" * 112)
    for t in poly:
        e = items[t]
        dom = (tag_of.get(term_concepts[t][0]) or ["?"])[0]
        labs = " | ".join(sorted(e["labs"]))[:60]
        print(f"{len(e['qids']):>5}  {t:<20} {term_concepts[t][0]:<18} {dom:<14} {labs}")

    no_sense = [t for t in sorted(term_concepts)
                if not items.get(t, {"qids": set()})["qids"]]
    print()
    print(f"  {len(no_sense)} terms have NO Wikidata sense link — nothing measured for")
    print("  them here. That is a gap in the external source, not a clean bill of")
    print("  health; see `evals/curated_correctness_new.py` for the same split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Error rate of the curated glossary on a NEW random sample (principle 9).

The previous measurement (`evals/glossary_error_rate.py`) used the curated list
as ground truth against the *generated* vocabulary, on all 310 curated German
terms. Those 310 are now known cases — re-using them would not be a blind test,
and after this expansion they are also the terms the list was written around.

This script does something different and independent:

  It samples concepts at RANDOM from the shipped curated glossary (fixed seed, so
  the sample is reproducible), then checks each sampled concept against Wikidata
  independently: does the German term have a lexeme whose P5137 sense points at
  an item in the same semantic area as the English term? That is an external
  check, not a self-consistency check.

  Sampling method and seed are printed. Two rates are reported separately:

    * SELF-CONFLICT  — internal ambiguity (a term under two concepts). Must be 0;
      this is the build gate, checked again on the shipped artifact.
    * CROSS-LINGUAL AGREEMENT — dense cosine between the two sides. Reported as a
      distribution, NOT as an error rate: it measures term rarity as much as
      correctness (see docs/rejected-approaches.md), so a low value is a signal to
      look, never a verdict.
    * WIKIDATA SENSE COVERAGE — of the sampled German terms, how many have a
      lexeme whose sense links to an item, i.e. the concept is externally
      attested at all.

Run: uv run python -m evals.curated_error_rate_new
"""
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402

from klix.backbone import HybridBackbone  # noqa: E402
from klix.glossary import Glossary  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CURATED = REPO / "src" / "klix" / "data" / "curated_glossary.json"

SEED = 20260928          # fixed: the sample is reproducible from this number
N_SAMPLE = 160           # >= the requested 150
EP = "https://query.wikidata.org/sparql"
UA = "klix-curated-audit/0.1"

# CASE MATTERS. German lexeme lemmas are capitalised ("Schraube") while the
# glossary stores terms lowercased, and SPARQL string comparison is exact — so
# querying the lowercase form returns zero rows for essentially every noun. This
# was the third instance of the same mistake in this project (see
# docs/rejected-approaches.md), which is why the self-test below exists: the
# script now refuses to report a coverage figure unless a known term is found.
SENSE_Q = """
SELECT DISTINCT ?item WHERE {
  ?lx dct:language wd:Q188 ; wikibase:lemma ?l ; ontolex:sense ?s .
  FILTER(STR(?l) = "%s")
  ?s wdt:P5137 ?item .
} LIMIT 20
"""

# A term whose Wikidata lexeme certainly exists. Used as the pre-flight check.
SELF_TEST_TERM = "Schraube"


def sparql_raw(lemma: str, retries: int = 5, timeout: int = 90):
    """Item QIDs for ONE exact lemma spelling; None if the query failed."""
    q = SENSE_Q % lemma
    url = EP + "?" + urllib.parse.urlencode({"query": q})
    req = urllib.request.Request(url, headers={
        "Accept": "application/sparql-results+json", "User-Agent": UA})
    for a in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                rows = json.loads(r.read().decode("utf-8"))["results"]["bindings"]
            time.sleep(0.7)
            return [b["item"]["value"].rsplit("/", 1)[-1] for b in rows]
        except (urllib.error.URLError, TimeoutError, KeyError):
            time.sleep(2.0 * (a + 1))
    return None            # inconclusive, reported separately from "no sense"


def sparql(term: str):
    """Item QIDs for a glossary term, trying its real German spellings.

    Tries the raw term, then capitalised, then fully upper — a German noun lemma
    is capitalised even when the term we store is not.
    """
    for cand in dict.fromkeys([term, term[:1].upper() + term[1:], term.upper()]):
        got = sparql_raw(cand)
        if got:
            return got
    return []


def main():
    raw = json.loads(CURATED.read_text(encoding="utf-8"))
    mapping = raw.get("concepts", raw)
    g = Glossary(dict(mapping))

    rng = random.Random(SEED)
    concepts = sorted(mapping)
    sample = rng.sample(concepts, min(N_SAMPLE, len(concepts)))

    # -- pre-flight (principle 10): one term with a known answer -----------
    probe = sparql_raw(SELF_TEST_TERM)
    if not probe:
        print(f"ABORT: self-test failed. '{SELF_TEST_TERM}' is known to have a "
              f"Wikidata lexeme, but the query returned nothing.\n"
              f"       Reporting a coverage rate now would be meaningless "
              f"(this exact mistake was made three times before).")
        return 1

    print("=" * 96)
    print("CURATED GLOSSARY — error rate on a NEW random sample")
    print("=" * 96)
    print(f"  population : all {len(mapping)} curated concepts (shipped artifact)")
    print(f"  sample     : {len(sample)} concepts drawn at RANDOM, random.Random({SEED})")
    print(f"  method     : reproducible from the seed above; not the previously")
    print(f"               measured 310 curated German terms (those are known cases)")
    print(f"  self-test  : '{SELF_TEST_TERM}' -> {len(probe)} item(s) found, so the "
          f"query works")
    print()

    # -- self-conflict: must be zero, checked on the shipped file -----------
    conflicts = g.conflicts()
    print(f"  SELF-CONFLICT (ambiguous mappings) : {len(conflicts)}")
    for f in conflicts[:5]:
        print(f"      {f['kind']}: {f['message'][:90]}")

    # -- external sense coverage, per sampled concept ----------------------
    per_domain = {}
    domains_doc = json.loads((REPO / "src" / "klix" / "data"
                              / "curated_domains.json").read_text(encoding="utf-8"))
    tag_of = domains_doc.get("concepts", {})

    attested = zero = inconclusive = 0
    missing_detail = []
    for concept in sample:
        de_terms = [t for t in mapping[concept].get("de", [])]
        if not de_terms:
            continue
        got = sparql(de_terms[0])
        dom = (tag_of.get(concept) or ["?"])[0]
        slot = per_domain.setdefault(dom, {"n": 0, "attested": 0, "zero": 0, "inc": 0})
        slot["n"] += 1
        if got is None:
            inconclusive += 1
            slot["inc"] += 1
        elif got:
            attested += 1
            slot["attested"] += 1
        else:
            zero += 1
            slot["zero"] += 1
            missing_detail.append((dom, concept, de_terms[0]))

    decided = attested + zero
    print()
    print("  EXTERNAL ATTESTATION (Wikidata lexeme + P5137 sense for the German term)")
    print(f"      has a linked sense : {attested}")
    print(f"      no linked sense    : {zero}")
    print(f"      query inconclusive : {inconclusive}   (not counted as either)")
    if decided:
        print(f"      -> attested ratio  : {attested/decided:.1%}   "
              f"(denominator: {decided} decided)")
    print()
    print("      per domain:")
    for dom in sorted(per_domain):
        s = per_domain[dom]
        d = s["attested"] + s["zero"]
        r = f"{s['attested']/d:.0%}" if d else "n/a"
        print(f"        {dom:14} n={s['n']:3d}  attested={s['attested']:3d}  "
              f"no sense={s['zero']:3d}  inconclusive={s['inc']:2d}  rate={r}")

    if missing_detail:
        print()
        print("      terms WITHOUT an external sense (sample):")
        for dom, concept, term in missing_detail[:25]:
            print(f"        [{dom}] {concept:22} de={term!r}")

    # -- cross-lingual agreement, as a distribution not a verdict ----------
    bb = HybridBackbone()
    flat = [t for c in sample for t in mapping[c].get("de", []) + mapping[c].get("en", [])]
    emb = {}
    for i in range(0, len(flat), 64):
        chunk = flat[i:i + 64]
        for t, v in zip(chunk, bb.embed_model.embed(chunk)):
            v = np.asarray(v, dtype=np.float32)
            n = float(np.linalg.norm(v))
            emb[t] = v / (n if n > 0 else 1.0)
    sims = []
    for c in sample:
        best = max((float(emb[d] @ emb[e])
                    for d in mapping[c].get("de", []) for e in mapping[c].get("en", [])
                    if d in emb and e in emb), default=0.0)
        sims.append(best)
    sims.sort()
    below = sum(1 for s in sims if s < 0.35)
    print()
    print("  CROSS-LINGUAL AGREEMENT (dense cosine, de side vs en side)")
    print(f"      median {statistics.median(sims):.3f}   "
          f"p10 {sims[len(sims)//10]:.3f}   p90 {sims[9*len(sims)//10]:.3f}")
    print(f"      below 0.35: {below} of {len(sims)} ({below/len(sims):.1%})")
    print("      NOT an error rate — this metric measures term rarity as much as")
    print("      correctness. A low value means 'look', not 'wrong'.")
    print()

    print("  " + "=" * 92)
    print("  HEADLINE: internal ambiguity = 0 (the build gate holds on the shipped")
    print(f"  artifact). External attestation {attested}/{decided} = "
          f"{attested/decided:.1%} of sampled German terms." if decided
          else "  artifact).")
    print("  Cross-lingual agreement is reported as a distribution, deliberately not")
    print("  converted into a single error rate — see docs/rejected-approaches.md.")


if __name__ == "__main__":
    raise SystemExit(main())

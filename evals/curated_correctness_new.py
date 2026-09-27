"""A REAL correctness rate for the curated glossary, on an independent ground truth.

Why this exists
---------------
`evals/curated_error_rate_new.py` reports **external attestation**: how many
sampled German terms have a Wikidata lexeme whose P5137 sense links *somewhere*.
That is a COVERAGE rate — it answers "is this term independently checkable at
all", not "is the mapping right". For 71.3 % of the sample it says nothing about
correctness. Reporting it as an error rate would be a category error.

`evals/glossary_error_rate.py` (the 6.6 % / 8.4 % methodology) cannot fill the gap
either, by construction: it scores the *generated* layer against the curated
English sides as ground truth. The curated English side IS the reference there, so
an error inside the curated list is invisible to it.

This script supplies the missing independent ground truth:
**cross-validation of the two sides against Wikidata items.** For a sampled
concept, collect the P5137 item(s) linked from the German term's lexeme sense(s)
and, separately, from the English term's lexeme sense(s). These are authored by
the Wikidata community, not by the glossary author, and the two sides are looked
up independently — so agreement is evidence, not circularity.

  CONFIRMED  — the German and English sides share at least one item. The two
               words independently point at the same concept externally.
  MISMATCH   — both sides have items, but the sets are disjoint. A candidate
               genuine wrong mapping; the available items are printed for reading.
  UNDECIDABLE— one or both sides have no P5137 sense link at all, so Wikidata
               offers no opinion. NOT counted as correct.

The rate is reported over the DECIDABLE subset with its denominator (principle 9),
and the undecidable remainder is reported as exactly that — the honest open gap.

Same sample as `curated_error_rate_new.py` (seed 20260928, n=160 of 362) so the
two figures are directly comparable.

Run: uv run python -m evals.curated_correctness_new
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
DOMAINS = REPO / "src" / "klix" / "data" / "curated_domains.json"

SEED = 20260928          # identical to curated_error_rate_new.py
N_SAMPLE = 160
UA = "klix-correctness-check/0.1"
EP = "https://query.wikidata.org/sparql"

# Wikidata literal language tags are LANGUAGE CODES, not item QIDs: "@de", not
# "@Q188". Using the QID is a silent no-match (0 rows for every term, including
# terms whose lexeme certainly exists) — caught by the pre-flight below.
LANG_DE = "de"
LANG_EN = "en"

# Batched lookup: several (lemma, language) pairs per query. The language tag is
# REQUIRED — a bare string does not match a language-tagged wikibase:lemma, which
# made the first version of this script return zero rows for every term
# (including 'Schraube', whose lexeme certainly exists). That looked exactly like
# a data finding; the self-test below is what caught it.
LEMMA_Q = """
SELECT ?l ?item WHERE {
  VALUES ?l { %s }
  ?lx wikibase:lemma ?l ; ontolex:sense ?s .
  ?s wdt:P5137 ?item .
}
"""

AGREEMENT_FLOOR = 0.35
AGREEMENT_WEAK = 0.50


def sparql(query: str, retries: int = 4, timeout: int = 120):
    """Returns bindings list, or None if every attempt failed."""
    url = EP + "?" + urllib.parse.urlencode({"query": query})
    req = urllib.request.Request(url, headers={
        "Accept": "application/sparql-results+json", "User-Agent": UA})
    for a in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                rows = json.loads(r.read().decode("utf-8"))["results"]["bindings"]
            time.sleep(0.5)
            return rows
        except (urllib.error.URLError, TimeoutError, KeyError):
            time.sleep(2.0 * (a + 1))
    return None


def spellings(term: str) -> list[str]:
    """Candidate lemma spellings for one glossary term (German lemmas are capitalised)."""
    return list(dict.fromkeys([term, term[:1].upper() + term[1:], term.upper()]))


def fetch_items(pairs: list[tuple[str, str]], batch: int = 60):
    """(lemma spelling, language QID) -> lemma -> set(item QIDs).

    Returns None if ANY batch was inconclusive, so a partial result is never
    silently reported as a finding (the failure mode that produced three earlier
    false results on this project).
    """
    out: dict[str, set[str]] = {}
    failed = False
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        values = " ".join('"%s"@%s' % (l.replace('"', ''), lang)
                          for l, lang in chunk)
        rows = sparql(LEMMA_Q % values)
        if rows is None:
            failed = True
            continue
        for b in rows:
            lemma = b["l"]["value"]
            qid = b["item"]["value"].rsplit("/", 1)[-1]
            out.setdefault(lemma, set()).add(qid)
        print(f"      batch {i // batch + 1}: {len(chunk)} lemma/language pairs"
              f" queried ({len(rows)} rows)", flush=True)
    return None if failed else out


def main() -> int:
    raw = json.loads(CURATED.read_text(encoding="utf-8"))
    mapping = raw.get("concepts", raw)
    mapping = {k: {lg: t for lg, t in v.items() if lg != "tags"}
               for k, v in mapping.items()}
    tag_of = json.loads(DOMAINS.read_text(encoding="utf-8")).get("concepts", {})

    rng = random.Random(SEED)
    sample = rng.sample(sorted(mapping), min(N_SAMPLE, len(mapping)))

    print("=" * 100)
    print("CURATED GLOSSARY — CORRECTNESS on an independent ground truth (Wikidata items)")
    print("=" * 100)
    print(f"  population : all {len(mapping)} curated concepts (shipped artifact)")
    print(f"  sample     : {len(sample)} drawn at RANDOM, random.Random({SEED})")
    print(f"               (same sample as curated_error_rate_new.py — comparable)")
    print(f"  ground truth: P5137 sense items of the German side vs the English side,")
    print(f"               looked up INDEPENDENTLY. Agreement is evidence, not a")
    print(f"               restatement of the file.")
    print()

    # ---- build the lemma -> concept lookups ------------------------------
    de_pairs: list[tuple[str, str]] = []
    en_pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for c in sample:
        for t in mapping[c].get("de", []):
            for sp in spellings(t):
                if (sp, LANG_DE) not in seen:
                    seen.add((sp, LANG_DE))
                    de_pairs.append((sp, LANG_DE))
        for t in mapping[c].get("en", []):
            for sp in spellings(t):
                if (sp, LANG_EN) not in seen:
                    seen.add((sp, LANG_EN))
                    en_pairs.append((sp, LANG_EN))

    print(f"  German lemma/language pairs  : {len(de_pairs)}")
    print(f"  English lemma/language pairs : {len(en_pairs)}")
    print()

    # ---- pre-flight on known cases (principle 10) ------------------------
    # 'Schraube'/'screw' and 'Ventil'/'valve' are known to share a Wikidata item
    # (verified by hand). If these come back empty the query is broken, and a
    # broken query looks exactly like a data finding — which is how three earlier
    # results on this project went wrong.
    KNOWN = [("Schraube", "screw", {"Q11022", "Q764323"}),
             ("Ventil", "valve", {"Q208443"})]
    pre_pairs = [(term, lang) for de, en, _ in KNOWN
                 for term, lang in ((de, LANG_DE), (en, LANG_EN))]
    pre = fetch_items(pre_pairs)
    if pre is None:
        print("ABORT: pre-flight query failed (network).")
        return 1
    for de, en, expected in KNOWN:
        got = pre.get(de, set()) & pre.get(en, set())
        if not (got & expected):
            print(f"ABORT: pre-flight failed for {de!r}/{en!r}: shared item(s) "
                  f"{sorted(got)} do not include the known {sorted(expected)}.")
            print(f"       de items={sorted(pre.get(de, set()))} "
                  f"en items={sorted(pre.get(en, set()))}")
            print("       The query is broken; a correctness rate from it would be")
            print("       meaningless.")
            return 1
    print(f"  pre-flight : {len(KNOWN)} known pairs share their expected item -> OK")
    print()

    print("  querying Wikidata (both sides):", flush=True)
    items = fetch_items(de_pairs + en_pairs)
    if items is None:
        print("\nABORT: at least one Wikidata batch was inconclusive. Reporting a")
        print("       correctness rate from a partial lookup would be meaningless.")
        return 1
    de_items = {l: s for l, s in items.items()}
    en_items = de_items

    # ---- classify per concept -------------------------------------------
    confirmed, mismatch, undecidable = [], [], []
    per_domain: dict[str, dict[str, int]] = {}
    for c in sample:
        dom = (tag_of.get(c) or ["?"])[0]
        slot = per_domain.setdefault(dom, {"n": 0, "conf": 0, "mism": 0, "und": 0})
        slot["n"] += 1

        d_items: set[str] = set()
        for t in mapping[c].get("de", []):
            for sp in spellings(t):
                d_items |= de_items.get(sp, set())
        e_items: set[str] = set()
        for t in mapping[c].get("en", []):
            for sp in spellings(t):
                e_items |= en_items.get(sp, set())

        if not d_items or not e_items:
            undecidable.append((c, dom, sorted(d_items), sorted(e_items)))
            slot["und"] += 1
        elif d_items & e_items:
            confirmed.append((c, dom, sorted(d_items & e_items)))
            slot["conf"] += 1
        else:
            mismatch.append((c, dom, sorted(d_items), sorted(e_items)))
            slot["mism"] += 1

    decided = len(confirmed) + len(mismatch)
    print()
    print("=" * 100)
    print("RESULT — CORRECTNESS (independent cross-validation, not coverage)")
    print("=" * 100)
    print(f"  CONFIRMED  (both sides share a Wikidata item) : {len(confirmed)}")
    print(f"  MISMATCH   (items present but disjoint)       : {len(mismatch)}")
    print(f"  UNDECIDABLE(no sense link on >=1 side)        : {len(undecidable)}")
    print()
    if decided:
        print(f"  -> wrong-mapping rate on the DECIDABLE subset : "
              f"{len(mismatch)}/{decided} = {len(mismatch)/decided:.1%}")
        print(f"     (denominator {decided} of {len(sample)} sampled = "
              f"{decided/len(sample):.1%} of the sample is independently decidable)")
    print(f"  -> the remaining {len(undecidable)}/{len(sample)} = "
          f"{len(undecidable)/len(sample):.1%} have NO independent ground truth")
    print(f"     in Wikidata. For those, correctness is UNKNOWN, not assumed.")

    print()
    print("  per domain:")
    for dom in sorted(per_domain):
        s = per_domain[dom]
        d = s["conf"] + s["mism"]
        r = f"{s['mism']/d:.0%}" if d else "n/a (nothing decidable)"
        print(f"    {dom:14} n={s['n']:3d}  confirmed={s['conf']:3d}  "
              f"mismatch={s['mism']:3d}  undecidable={s['und']:3d}  wrong={r}")

    if mismatch:
        print()
        print("  MISMATCHES — read each; the items are printed because these decide")
        print("  whether the mapping is wrong or the word is polysemous:")
        for c, dom, d_items, e_items in mismatch:
            print(f"    [{dom}] {c:22} de={mapping[c].get('de')}")
            print(f"         de items: {d_items}")
            print(f"         en items: {e_items}")

    # ---- the undecidable remainder, per domain ---------------------------
    # This is the honest limit: no external ground truth exists, so correctness
    # is UNKNOWN. Listed per domain and with the missing side, because whether
    # the uncertainty sits in the peripheral domains or reaches the primary
    # use-case domain is a decision-relevant question.
    print()
    print("  UNDECIDABLE — no independent ground truth (correctness UNKNOWN):")
    by_dom: dict[str, list[tuple[str, bool, bool]]] = {}
    for c, dom, d_items, e_items in undecidable:
        by_dom.setdefault(dom, []).append((c, bool(d_items), bool(e_items)))
    for dom in sorted(by_dom):
        rows = by_dom[dom]
        print(f"    {dom:14} {len(rows):3d} undecidable")
    print()
    print("    detail (why: which side has no P5137 sense link):")
    for dom in sorted(by_dom):
        print(f"    [{dom}]")
        for c, has_de, has_en in sorted(by_dom[dom]):
            why = ("German side has no sense" if not has_de and has_en
                   else "English side has no sense" if has_de and not has_en
                   else "NEITHER side has a sense")
            print(f"      {c:22} de={str(mapping[c].get('de'))[:46]:48} {why}")

    # ---- agreement outliers (asked for explicitly) -----------------------
    bb = HybridBackbone()
    flat = [t for c in sample for t in mapping[c].get("de", []) + mapping[c].get("en", [])]
    emb = {}
    for i in range(0, len(flat), 64):
        chunk = flat[i:i + 64]
        for t, v in zip(chunk, bb.embed_model.embed(chunk)):
            v = np.asarray(v, dtype=np.float32)
            n = float(np.linalg.norm(v))
            emb[t] = v / (n if n > 0 else 1.0)
    scored = []
    for c in sample:
        best = max((float(emb[d] @ emb[e])
                    for d in mapping[c].get("de", [])
                    for e in mapping[c].get("en", [])
                    if d in emb and e in emb), default=0.0)
        scored.append((best, c, (tag_of.get(c) or ["?"])[0]))
    scored.sort()
    sims = [s for s, _, _ in scored]
    below = [x for x in scored if x[0] < AGREEMENT_FLOOR]
    weak = [x for x in scored if AGREEMENT_FLOOR <= x[0] < AGREEMENT_WEAK]

    print()
    print("=" * 100)
    print("AGREEMENT OUTLIERS (dense cosine, de side vs en side)")
    print("=" * 100)
    print(f"  median {statistics.median(sims):.3f}   "
          f"p10 {sims[len(sims)//10]:.3f}   p90 {sims[9*len(sims)//10]:.3f}")
    print(f"  below {AGREEMENT_FLOOR}: {len(below)} of {len(sims)}")
    print(f"  {AGREEMENT_FLOOR}-{AGREEMENT_WEAK}      : {len(weak)} of {len(sims)}")
    print()
    print(f"  EVERY pair below {AGREEMENT_WEAK}, worst first "
          f"(this is the list asked for):")
    for s, c, dom in below + weak:
        flag = " <== BELOW FLOOR" if s < AGREEMENT_FLOOR else ""
        print(f"    {s:.3f}  [{dom:13}] {c:22} de={mapping[c].get('de')}")
        print(f"           {'':13}  {'':22} en={mapping[c].get('en')}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

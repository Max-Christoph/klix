"""Spot-check the multilingual core against an independent source (Wikidata).

Why this exists
---------------
The 25x10 multilingual pack is hand-written, single-author, MIT. "Hand-written"
is a provenance claim, not a correctness claim — the curated manufacturing review
found genuine errors in hand-written terms, and 8 of these 10 languages are ones
the author does not speak natively. This script is the agreed sampling check:
~12 terms per NEW language (fr, es, it, pt, nl, pl, sv, da — de/en are the
author's own and covered by the curated review methodology), validated against a
second, independent source.

Method (and why the previous attempts failed)
----------------------------------------------
* SPARQL (the curated review's route): refused — active WDQS outage, hard 429
  rate limit of 1 req/min. Every one of 96 queries returned garbage; that run
  was discarded, not reported as "0 mismatches".
* Lexeme sense items (P5137) via REST: runs, but the top-5 lemma hits carry no
  usable P5137 links for this vocabulary — 0/25 English references resolved, so
  every verdict collapses to UNDECIDABLE. A check that cannot confirm anything
  is not a check.
* First item-label attempt: the Q-IDs were written from memory and 22 of 24
  were WRONG ("invoice" resolved to Hugh IV, Duke of Burgundy). The IDs below
  were then re-resolved through `wbsearchentities` and every hit was READ
  before mapping. This is noted here because it is the exact failure mode the
  spot check exists to catch — in the author's own work, not just the pack.

What this script does — item-label cross-validation:

  1. Each concept maps to a Wikidata item (table below; every ID re-resolved via
     `wbsearchentities` and read, not typed from memory).
  2. `wbgetentities(props=labels)` returns that item's labels in the 10 pack
     languages — authored by the Wikidata community, independently of this pack.
  3. A term in language L is CONFIRMED when it matches the item's L-label
     (case-insensitive; head-noun match in both directions, because labels are
     dictionary forms while pack terms include inflected/compound variants).

  CONFIRMED  — term and community label name the same thing.
  MISMATCH   — item has a label in that language, term does not match it.
               Printed for manual reading: could be a wrong term, a string-match
               miss on a legitimate inflection, or a real synonym.
  UNDECIDABLE— item carries no label in that language: no opinion. NOT counted.

This is weaker than sense disambiguation (a label match cannot prove the term is
not ALSO something else) but symmetric, independent, and every MISMATCH is
printed for human reading. The item mapping is part of the script so the item
choice itself is reviewable.

Run: uv run python -m evals.multilingual_spotecheck

Result of the completed run (2026-09-29, seed 20260929, 12/language, WITH the
accent fold): 37 CONFIRMED, 54 MISMATCH, 5 UNDECIDABLE (40.7 % over decidable).

The 54 MISMATCHes were classified by reading every one:

  ~32  sampled term is a correct synonym/inflection whose SIBLING in the same
       pack concept matches the label (fr error: sampled 'panne', pack also has
       'erreur' = the label). A label match structurally cannot confirm a
       synonym — this is the method's ceiling, not a pack error.
  ~22  sense/register differences vs. the Wikidata item, all legitimate
       (training = education vs. Wikidata's sport sense; review = assessment
       vs. book critique; question = request vs. the linguistics item).

  0    confirmed wrong mappings in the sample. The check surfaced candidates;
       every candidate was read; none was wrong.

Honest gaps, stated and not smoothed over:
  * 10 of 25 concepts (urgent, cancel, help, approve, reject, access, status,
    escalate, cost, delivery) have no clean single Wikidata item — homonyms
    won the search (the Foreigner single, the Carpenters track) — and are NOT
    externally checked. Full sense disambiguation would need SPARQL, which was
    in a hard outage during this check.
  * A label match proves the term names the same thing as the item; it cannot
    prove the term is not ALSO something else (polysemy stays unmeasured).
  * This is a sample (96 of 250 entries), not a full audit.
"""
import json
import random
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, "src")
sys.path.insert(0, ".")

NEW_LANGS = ("fr", "es", "it", "pt", "nl", "pl", "sv", "da")
ALL_LANGS = ("da", "de", "en", "es", "fr", "it", "nl", "pl", "pt", "sv")
PER_LANG = 12
SEED = 20260929
PACE = 1.5
UA = {"User-Agent": "klix-eval/0.10 (glossary spot check; repo: klix)"}

# Concept -> Wikidata item. Every ID was resolved via wbsearchentities and the
# hit's label+description READ before being mapped (several search results are
# homonyms: "urgent" the Foreigner single, "help" the Carpenters track, "quality"
# the Talib Kweli album — none of them mapped). Concepts without a clean single
# item are excluded on purpose and listed as NOT CHECKED, not silently skipped.
CONCEPT_ITEM = {
    "conveyor": "Q770135",      # belt conveyor
    "maintenance": "Q1043452",   # maintenance
    "error": "Q29485",          # error (inaccurate action/proposition)
    "bug": "Q179550",           # software bug
    "invoice": "Q190581",       # invoice
    "refund": "Q110087232",     # refund (return of payment)
    "question": "Q189756",      # question (request for information)
    "change": "Q1150070",       # change (deviation from present state)
    "incident": "Q12890393",    # incident (event/occurrence)
    "review": "Q265158",        # review (evaluation)
    "deadline": "Q1465133",     # time limit
    "request": "Q22445448",     # request (act of asking)
    "training": "Q918385",      # training
    "report": "Q10870555",      # report (informational text)
    "quality": "Q1207505",      # quality (distinguishing feature)
    "safety": "Q10566551",      # safety (state of being secure)
    # NOT CHECKED (no clean single item found by reading the search results):
    #   urgent, cancel, help, approve, reject, access, status, escalate, cost,
    #   delivery — homonymous or too abstract; would need sense disambiguation.
    # They remain covered by the pack's own validate() (structural checks) but
    # NOT by this external check. That is an honest gap, not a pass.
}
NOT_CHECKED = {"urgent", "cancel", "help", "approve", "reject", "access",
               "status", "escalate", "cost", "delivery"}


def api(params: dict, retries: int = 4) -> dict | None:
    params = dict(params, format="json")
    url = "https://www.wikidata.org/w/api.php?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries - 1:
                time.sleep((10, 30, 60)[min(attempt, 2)])
                continue
            print(f"    (request failed: {exc})")
            return None
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                print(f"    (request failed: {exc})")
                return None
            time.sleep(5 * (attempt + 1))
    return None


def item_labels(qid: str) -> dict[str, str]:
    data = api({"action": "wbgetentities", "ids": qid, "props": "labels",
                "languages": "|".join(ALL_LANGS)})
    if data is None:
        return {}
    ent = data.get("entities", {}).get(qid, {})
    return {lg: v["value"] for lg, v in ent.get("labels", {}).items()}


def _norm(s: str) -> str:
    """Lowercase, punctuation-stripped, accent-folded (NFD -> ASCII).

    The accent fold is deliberate and was added after the first run reported 39
    of 61 "mismatches" that were fold-variants of confirmed terms ("qualite" vs
    "qualité", "spoergsmaal" vs "spørgsmål"). klix packs carry BOTH spellings on
    purpose — queries arrive both ways — so a checker that does not fold calls
    half the pack wrong. The fold is used for MATCHING only; the pack data keeps
    both forms.
    """
    lowered = s.lower().strip(" .,;:!?")
    return "".join(c for c in unicodedata.normalize("NFD", lowered)
                   if unicodedata.category(c) != "Mn")


def matches(term: str, label: str) -> bool:
    """Whole-word / head-noun match in both directions, accent-folded."""
    t, l = _norm(term), _norm(label)
    if not t or not l:
        return False
    if t == l or t in l.split() or l in t.split():
        return True
    # head noun of one side vs the whole of the other
    # ("cinta transportadora" ~ "transportadora"; "aflysning" ~ "aflys")
    head_l = l.split()[-1]
    head_t = t.split()[-1]
    return head_l == t or head_t == l or head_l == head_t


def main() -> None:
    rng = random.Random(SEED)
    from klix.glossaries import multilingual
    pack = multilingual().mapping

    print("=" * 88)
    print(f"MULTILINGUAL PACK SPOT CHECK — item-label cross-validation, seed {SEED}")
    print("independent source: Wikidata item labels (community-authored)")
    print(f"concepts with a clean item: {len(CONCEPT_ITEM)}; "
          f"NOT CHECKED (no clean item): {sorted(NOT_CHECKED)}")
    print("=" * 88)

    labels: dict[str, dict[str, str]] = {}
    for concept, qid in CONCEPT_ITEM.items():
        labels[concept] = item_labels(qid)
        time.sleep(PACE)

    total = {"CONFIRMED": 0, "MISMATCH": 0, "UNDECIDABLE": 0}
    per_lang = {lg: [0, 0, 0] for lg in NEW_LANGS}  # confirmed, mismatch, undecidable
    findings = []

    for lang in NEW_LANGS:
        candidates = []
        for concept, langs in pack.items():
            if concept not in CONCEPT_ITEM:
                continue
            for term in langs.get(lang, []):
                candidates.append((concept, term))
        rng.shuffle(candidates)
        sample = candidates[:PER_LANG]
        print(f"\n{lang}: {len(sample)} sampled")
        for concept, term in sample:
            lab = labels.get(concept, {}).get(lang)
            if not lab:
                verdict = "UNDECIDABLE"
            elif matches(term, lab):
                verdict = "CONFIRMED"
            else:
                verdict = "MISMATCH"
                findings.append((lang, concept, term, lab))
            total[verdict] += 1
            per_lang[lang][{"CONFIRMED": 0, "MISMATCH": 1, "UNDECIDABLE": 2}[verdict]] += 1
            mark = f"   (label: {lab!r})" if verdict == "MISMATCH" else ""
            print(f"  {verdict:11} {term:28} [{concept}]{mark}")

    decided = total["CONFIRMED"] + total["MISMATCH"]
    print("\n" + "=" * 88)
    print("SUMMARY")
    for key in ("CONFIRMED", "MISMATCH", "UNDECIDABLE"):
        print(f"  {key:11} {total[key]}")
    if decided:
        print(f"  confirmed rate over DECIDABLE: {total['CONFIRMED']}/{decided}"
              f" = {100 * total['CONFIRMED'] / decided:.1f}%")
    for lang, (c, m, u) in per_lang.items():
        print(f"    {lang}: {c} confirmed, {m} mismatch, {u} undecidable")
    if findings:
        print("\nMISMATCHES for manual reading (pack term vs community label):")
        for lang, concept, term, lab in findings:
            print(f"  {lang}/{concept}: pack {term!r} vs label {lab!r}")
    else:
        print("\nno mismatches in the sample")
    print("\ncoverage note: concepts without a clean Wikidata item were NOT "
          "externally checked (listed above); this is a sample check, not a "
          "full audit of all 250 entries.")


if __name__ == "__main__":
    main()
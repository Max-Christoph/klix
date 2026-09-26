#!/usr/bin/env python
"""Generate klix's bundled default glossary (broad DE<->EN basic vocabulary).

Why this script exists
----------------------
klix routes multilingual text through a *flat* concept lookup: word -> concept
-> synonyms. The bundled vocabulary started as 16 hand-written manufacturing
terms; a broad basic vocabulary (10k-25k terms) is what makes the keyword
channel useful for arbitrary support/IT/office text instead of only shop-floor
language.

Data source, and why not OMW / MUSE
-----------------------------------
The obvious candidate was NLTK's Open Multilingual WordNet (OMW 1.4). Two hard
blockers, both checked on 2026-09:

1. NLTK's ``omw-1.4`` ships **no German data at all** (there is no ``deu``
   directory; only als/arb/bul/cow/dan/ell/fin/fra/heb/hrv/isl/ita/iwn/jpn/mcr/
   msa/nld/nor/pol/por/ron/slk/slv/swe/tha). A DE<->EN dictionary cannot be
   built from it.
2. The OMW language sets that do exist are mostly **share-alike** (CC BY-SA for
   nld/por/slv, CC BY 3.0 for swe/fin, CeCILL-C for fra, ...). Bundling
   share-alike data inside an MIT-licensed package is a licence conflict.

Chosen source: **Wikidata** (CC0 1.0 — public-domain dedication, no attribution
or share-alike obligation). Two complementary routes, both high-precision:

* **Lexemes** (``wikibase:Lexeme``, route A — primary). A German lexeme carries
  a real part-of-speech tag (``wikibase:lexicalCategory``: noun Q1084, verb
  Q24905, adjective Q34698) and links through ``wdt:P5137`` (item for this
  sense) to an English lexeme with the same meaning. This is a genuine
  translation dictionary, POS-filtered by construction, and free of the
  proper-noun noise that Wikidata *item* labels carry.
* **Item labels in curated content classes** (route B, supplementary). German
  and English labels of the same item, restricted to class subtrees that are
  word-like (tools, machines, materials, processes, qualities, units, foods,
  animals, body parts, colours, ...). This adds general vocabulary the lexeme
  route does not cover, while an explicit class allow-list keeps names, people,
  places and works out.

Output
------
``src/klix/data/default_glossary.json`` in the standard klix multi-language
format::

    {"CONCEPT_KEY": {"de": [...], "en": [...], "fr": [...]}}

The concept key is the ASCII-folded, lowercased English label with non-word
characters collapsed to ``_`` — deterministic and human-readable, so the file
stays diffable and reviewable. Any number of ISO-639-1 keys per concept is
allowed; this build emits ``de`` and ``en``.

Determinism
-----------
Every query carries ``ORDER BY``, so pagination is stable across runs (verified:
two identical limit/offset runs return byte-identical rows). Raw responses are
cached under ``.hermes/glossary_cache/`` keyed by a hash of the query, so
re-running the build is offline and produces the identical file.

Licence
-------
Wikidata content is CC0 1.0. No attribution is required; ``DATA_SOURCES.md``
credits it anyway, and names the licence chain of every rejected alternative.

Run: uv run python scripts/build_default_glossary.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "klix-glossary-build/0.1 (https://github.com/Max-Christoph/klix)"
OUT_PATH = REPO / "src" / "klix" / "data" / "default_glossary.json"
CACHE_DIR = REPO / ".hermes" / "glossary_cache"

# Wikidata entity ids
Q_DE = "Q188"        # German language
Q_EN = "Q1860"       # English language
POS_CONTENT = ["Q1084", "Q24905", "Q34698"]  # noun, verb, adjective

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

# Class subtrees for route B. Deliberately word-like: each entry was picked so
# that its instances are things/qualities/processes people write tickets about.
CONTENT_CLASSES: list[tuple[str, str]] = [
    ("tools",              "Q39546"),
    ("machines",           "Q11019"),
    ("materials",          "Q35758"),
    ("substances",         "Q79529"),
    ("processes",          "Q1914636"),
    ("qualities",          "Q1207505"),
    ("units",              "Q47574"),
    ("computer_terms",     "Q21146257"),
    ("foods",              "Q2095"),
    ("drinks",             "Q40050"),
    ("animals",            "Q729"),
    ("plants",             "Q756"),
    ("body_parts",         "Q4936952"),
    ("colors",             "Q1075"),
    ("weather",            "Q11663"),
    ("emotions",           "Q9415"),
    ("vehicles",           "Q42889"),
    ("clothing",           "Q11460"),
    ("furniture",          "Q14745"),
    ("buildings",          "Q41176"),
    ("occupations",        "Q12737077"),
    ("diseases",           "Q12136"),
    ("chemical_elements",  "Q11344"),
    ("minerals",           "Q7946"),
    ("textiles",           "Q28823"),
    ("sports",             "Q349"),
    ("instruments",        "Q34379"),
    ("shapes",             "Q16101919"),
    ("measures",           "Q12453"),
    ("directions",         "Q2151613"),
]

NAME_LIKE_RE = re.compile(
    r"\b("
    r"university|institute|company|corporation|gmbh|ltd|inc|plc|foundation|"
    r"association|party|club|team|airport|station|museum|hospital|school|"
    r"album|song|film|movie|series|novel|magazine|newspaper|journal|"
    r"county|province|district|municipality|village|town|city|river|mountain|"
    r"war|battle|treaty|republic|kingdom|empire|dynasty|province"
    r")\b",
    re.IGNORECASE,
)


def _stopwords() -> set[str]:
    """Stopword filter (DE + EN), derived from the engine's own list."""
    from klix.backbone import _DEFAULT_STOPWORDS

    return {w.lower() for w in _DEFAULT_STOPWORDS}


# Function words the backbone's compact list does not carry. Articles,
# pronouns, auxiliaries, prepositions, particles — never routing signal.
FUNCTION_WORD_EXTRA = {
    "welche", "welcher", "welches", "dieser", "diese", "dieses", "jener", "jene",
    "mein", "meine", "meiner", "dein", "deine", "sein", "seine", "ihre", "unser",
    "wird", "werden", "wurde", "wurden", "worden", "kann", "können", "muss",
    "müssen", "soll", "sollen", "darf", "dürfen", "mag", "mögen", "würde",
    "würden", "hätte", "hätten", "wäre", "wären", "gewesen", "bin", "bist",
    "seid", "ohne", "gegen", "durch", "wegen", "trotz", "statt", "während",
    "innerhalb", "außerhalb", "damit", "dafür", "daran", "darauf", "darin",
    "davon", "dazu", "dabei", "dadurch", "deshalb", "deswegen", "daher",
    "trotzdem", "obwohl", "sobald", "solange", "sowie", "sowohl", "weder",
    "entweder", "jedenfalls", "jeweils", "meistens", "manchmal", "immer", "nie",
    "niemals", "oft", "selten", "kaum", "fast", "etwa", "circa", "sehr",
    "ziemlich", "ganz", "völlig", "überhaupt", "eigentlich", "bereits", "erst",
    "noch", "schon", "gerade", "eben", "halt", "mal", "doch", "denn", "also",
    "whos", "whose", "whom", "which", "ourselves", "yourselves", "themselves",
    "itself", "myself", "yourself", "himself", "herself", "ours", "yours",
    "hers", "theirs", "having", "doing", "done", "ought", "unto", "upon",
    "within", "without", "across", "along", "among", "around", "behind",
    "below", "beneath", "beside", "between", "beyond", "during", "except",
    "inside", "outside", "through", "throughout", "toward", "towards",
    "underneath", "until", "unless", "whether", "however", "therefore", "thus",
    "hence", "moreover", "furthermore", "nevertheless", "otherwise", "indeed",
    "perhaps", "maybe", "rather", "quite", "almost", "nearly", "barely",
    "hardly", "merely", "simply", "entirely", "completely", "absolutely",
    "something", "anything", "nothing", "everything", "someone", "anyone",
    "everyone", "somebody", "anybody", "everybody", "somewhere", "anywhere",
    "everywhere", "nowhere", "etwas", "nichts", "alles", "jemand", "niemand",
    "irgendwas", "irgendwer", "man", "leute",
    # Interjections / greetings (no routing signal, but frequent in tickets)
    "hallo", "hi", "hey", "moin", "servus", "grüße", "grüsse", "danke", "bitte",
    "hello", "thanks", "thank", "please", "regards", "cheers",
}


def ascii_fold(s: str) -> str:
    """ä->a, ß->ss, ... (NFKD + strip combining marks), lowercased."""
    s = s.replace("ß", "ss").replace("ẞ", "ss")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def clean_term(s: str, min_len: int = 3, max_tokens: int = 3,
               stop: set[str] | None = None) -> str | None:
    """Normalizes one label; returns None when it must be dropped.

    Drops trailing disambiguators ("Belt (mechanical)"), numbers, punctuation,
    abbreviations shorter than `min_len` (prevents acronym false positives),
    anything longer than `max_tokens` words (a label that long is a description,
    not a lookup key) and any multi-word label that STARTS with a function word
    ("a map of japan", "the united states", "eine karte japans") — those are
    phrases, not vocabulary, and they inject noise into the sparse channel.
    """
    s = (s or "").strip()
    if not s:
        return None
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = s.split()
    if len(tokens) > max_tokens:
        return None
    if any(ch.isdigit() for ch in s):
        return None
    if re.search(r"[^\w\s\-']", s, flags=re.UNICODE):  # punctuation except - and '
        return None
    if "http" in s.lower() or "@" in s:
        return None
    if NAME_LIKE_RE.search(s):
        return None
    if stop and len(tokens) > 1 and tokens[0].lower() in stop:
        return None
    if len(ascii_fold(s)) < min_len:
        return None
    return s.lower()


def concept_key(en_term: str) -> str:
    """Deterministic, readable concept key from the English term."""
    key = ascii_fold(en_term)
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return key


# ---------------------------------------------------------------------------
# SPARQL, with an on-disk cache
# ---------------------------------------------------------------------------

def sparql(query: str, retries: int = 3, timeout: int = 240) -> list[dict]:
    """Runs one SPARQL query with caching + retry; returns the bindings."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query})
    req = urllib.request.Request(url, headers={
        "Accept": "application/sparql-results+json",
        "User-Agent": USER_AGENT,
    })
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                bindings = json.loads(resp.read().decode("utf-8"))["results"]["bindings"]
            cache_file.write_text(json.dumps(bindings, ensure_ascii=False), encoding="utf-8")
            return bindings
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"SPARQL failed after {retries} attempts: {last}")


LEXEME_QUERY = """
SELECT ?de ?en ?pos WHERE {{
  ?lde dct:language wd:{qde}; wikibase:lemma ?de;
       ontolex:sense ?sde; wikibase:lexicalCategory ?pos .
  VALUES ?pos {{ wd:{pos} }}
  ?sde wdt:P5137 ?item .
  ?len dct:language wd:{qen}; ontolex:sense ?sen; wikibase:lemma ?en .
  ?sen wdt:P5137 ?item .
  FILTER(LANG(?de) = "de" && LANG(?en) = "en")
}}
ORDER BY ?de ?en
LIMIT {limit} OFFSET {offset}
"""

ITEM_QUERY = """
SELECT ?de ?en WHERE {{
  ?item wdt:P31/wdt:P279* wd:{cls} .
  ?item rdfs:label ?de . FILTER(LANG(?de) = "de")
  ?item rdfs:label ?en . FILTER(LANG(?en) = "en")
}}
ORDER BY ?item
LIMIT {limit} OFFSET {offset}
"""


def fetch_lexemes(limit: int, offset: int) -> list[dict]:
    q = LEXEME_QUERY.format(qde=Q_DE, qen=Q_EN, pos=" wd:".join(POS_CONTENT),
                            limit=limit, offset=offset)
    return sparql(q)


def fetch_item_class(cls: str, limit: int, offset: int,
                     retries: int = 3, timeout: int = 240) -> list[dict]:
    return sparql(ITEM_QUERY.format(cls=cls, limit=limit, offset=offset),
                  retries=retries, timeout=timeout)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

class Assembler:
    """Collects DE/EN pairs into concepts, deterministically and conflict-free.

    Invariant: one surface word resolves to exactly ONE concept. When a new pair
    arrives whose German term is already owned by another concept (a homograph),
    the pair is dropped rather than silently re-pointing an existing key — a
    glossary lookup can only return one concept, so an ambiguous entry is worse
    than a missing one. The drop count is reported at the end.
    """

    def __init__(self, max_terms: int, domain: dict | None = None):
        self.out: dict[str, dict[str, list[str]]] = {}
        # ONE owner per surface term, across BOTH languages. The klix lookup is
        # flat and language-agnostic (`word -> concept`), so a term must not be
        # registered under two concepts — not even as German in one and English
        # in another. Tracking owners per language allowed exactly that and
        # produced 287 ambiguous entries in the first generated file.
        self.owner: dict[str, str] = {}
        self.max_terms = max_terms
        self.stats = {"added": 0, "dropped_conflict": 0, "dropped_filter": 0,
                      "merged_synonym": 0}
        for concept, langs in (domain or {}).items():
            self._create(concept, langs.get("de", []), langs.get("en", []))

    def _create(self, key: str, de_terms: list[str], en_terms: list[str]) -> None:
        bucket = self.out.setdefault(key, {"de": [], "en": []})
        for t in de_terms:
            if t not in bucket["de"] and len(bucket["de"]) < self.max_terms:
                bucket["de"].append(t)
                self.owner.setdefault(t, key)
        for t in en_terms:
            if t not in bucket["en"] and len(bucket["en"]) < self.max_terms:
                bucket["en"].append(t)
                self.owner.setdefault(t, key)

    def add(self, de_raw: str, en_raw: str, stop: set[str]) -> bool:
        de = clean_term(de_raw, stop=stop)
        en = clean_term(en_raw, stop=stop)
        if not de or not en or de in stop or en in stop:
            self.stats["dropped_filter"] += 1
            return False
        if de == en:  # cognate/identical: no bridging value
            self.stats["dropped_filter"] += 1
            return False

        # The concept key is derived from the English term; if that key is
        # already taken by a DIFFERENT concept the pair is ambiguous too.
        key = concept_key(en)
        if len(key) < 3:
            self.stats["dropped_filter"] += 1
            return False

        known_de = self.owner.get(de)
        known_en = self.owner.get(en)
        known_key = self.owner.get(key)
        owners = {o for o in (known_de, known_en, known_key) if o is not None}
        if len(owners) > 1:
            self.stats["dropped_conflict"] += 1
            return False
        concept = owners.pop() if owners else None
        if concept is None:
            self._create(key, [de], [en])
            self.stats["added"] += 1
            return True

        # Extend an existing concept with the new synonym (capped).
        bucket = self.out[concept]
        grew = False
        if de not in bucket["de"] and len(bucket["de"]) < self.max_terms:
            bucket["de"].append(de)
            self.owner.setdefault(de, concept)
            grew = True
        if en not in bucket["en"] and len(bucket["en"]) < self.max_terms:
            bucket["en"].append(en)
            self.owner.setdefault(en, concept)
            grew = True
        if grew:
            self.stats["merged_synonym"] += 1
        return grew

    def finalize(self) -> dict[str, dict[str, list[str]]]:
        """Drops concepts missing one side; sorts everything deterministically."""
        out: dict[str, dict[str, list[str]]] = {}
        for key in sorted(self.out):
            langs = self.out[key]
            de = sorted({t for t in langs.get("de", [])})
            en = sorted({t for t in langs.get("en", [])})
            if not de or not en:
                continue          # a concept needs both sides to bridge
            out[key] = {"de": de, "en": en}
        return out


def build(target_concepts: int, page: int, domain: dict, max_terms: int,
          use_items: bool, item_page: int = 800, per_class: int = 4000,
          quiet: bool = False):
    stop = _stopwords() | FUNCTION_WORD_EXTRA
    asm = Assembler(max_terms, domain=domain)

    def log(*a):
        if not quiet:
            print(*a, flush=True)

    log(f"  domain pack seeded : {len(asm.out)} concepts")

    # Route A: POS-tagged lexemes (the precise translation-dictionary core).
    offset = 0
    while len(asm.out) < target_concepts:
        batch = fetch_lexemes(page, offset)
        if not batch:
            break
        for b in batch:
            asm.add(b["de"]["value"], b["en"]["value"], stop)
        offset += page
        log(f"  [lexemes] offset {offset:6d}  concepts {len(asm.out):6d}  "
            f"terms {sum(len(t) for v in asm.out.values() for t in v.values()):6d}")
        if len(batch) < page:
            break

    # Route B: curated content classes via item labels.
    if use_items and len(asm.out) < target_concepts:
        for name, cls in CONTENT_CLASSES:
            if len(asm.out) >= target_concepts:
                break
            offset = 0
            scanned = 0
            try:
                while len(asm.out) < target_concepts and scanned < per_class:
                    batch = fetch_item_class(cls, item_page, offset, retries=1, timeout=75)
                    if not batch:
                        break
                    for b in batch:
                        asm.add(b["de"]["value"], b["en"]["value"], stop)
                    scanned += len(batch)
                    offset += item_page
                    if len(batch) < item_page:
                        break
            except RuntimeError as exc:
                # A deep class subtree can exceed the query service's timeout
                # (WDQS returns 504). That is not a build failure: the lexeme
                # route and the other classes already carry the vocabulary.
                log(f"  [{name:16s}] SKIPPED (query service) — {exc}")
                continue
            log(f"  [{name:16s}] concepts {len(asm.out):6d}  scanned {scanned:5d}  "
                f"terms {sum(len(t) for v in asm.out.values() for t in v.values()):6d}")

    return asm.finalize(), asm.stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=20000,
                    help="stop once this many concepts are collected (default 20000)")
    ap.add_argument("--page", type=int, default=2500, help="SPARQL page size")
    ap.add_argument("--max-terms", type=int, default=4,
                    help="max synonyms per language per concept (default 4)")
    ap.add_argument("--no-items", action="store_true",
                    help="lexeme route only (skip the curated item classes)")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    # Domain words come from the code presets: single source of truth.
    from klix.glossaries import MANUFACTURING, WORKFLOW

    domain: dict[str, dict[str, list[str]]] = {}
    for src in (MANUFACTURING, WORKFLOW):
        for concept, langs in src.items():
            bucket = domain.setdefault(concept, {"de": [], "en": []})
            for lg, terms in langs.items():
                bucket[lg] = list(dict.fromkeys(bucket.get(lg, []) + [t.lower() for t in terms]))

    t0 = time.perf_counter()
    print("Building klix default glossary")
    print("  source : Wikidata (CC0 1.0) — lexemes (POS-tagged) + curated item classes")
    data, stats = build(args.target, args.page, domain, args.max_terms,
                        use_items=not args.no_items, quiet=args.quiet)
    dt = time.perf_counter() - t0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")), encoding="utf-8")
    n_terms = sum(len(t) for v in data.values() for t in v.values())
    size_bytes = args.out.stat().st_size
    size_mb = size_bytes / 1048576
    size_str = f"{size_mb:.2f} MB" if size_mb >= 0.01 else f"{size_bytes / 1024:.0f} KB"

    print()
    print(f"  pairs added          : {stats['added']}")
    print(f"  synonyms merged in   : {stats['merged_synonym']}")
    print(f"  dropped (homograph)  : {stats['dropped_conflict']}")
    print(f"  dropped (filter)     : {stats['dropped_filter']}")
    print(f"  concepts             : {len(data)}")
    print(f"  terms (de+en)        : {n_terms}")
    print(f"  file                 : {args.out}  ({size_str})")
    print(f"  budget               : {'OK' if size_mb <= 2.0 else 'OVER'} vs 2 MB target")
    print(f"  build time           : {dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

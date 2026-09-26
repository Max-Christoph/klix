# Data sources and licences

klix ships one generated data file: `src/klix/data/default_glossary.json`.
Everything else in the package is code. This file records where that data comes
from, why the alternatives were rejected, and how to regenerate it.

## Shipped data

| File | Content | Source | Licence |
|---|---|---|---|
| `src/klix/data/default_glossary.json` | Broad DE↔EN basic vocabulary (concepts → per-language synonyms), generated offline | [Wikidata](https://www.wikidata.org) | **CC0 1.0** (public domain dedication) |
| `src/klix/glossary.json` | 16-term manufacturing/OT production glossary, hand-written | klix itself | MIT (same as the package) |

Both files are consumed read-only at runtime. Nothing is downloaded at install
time or at import time — `pip install klix-engine` is fully offline.

`default_glossary.json` is **generated**, not hand-maintained. Regenerate with:

```bash
uv run python scripts/build_default_glossary.py
```

The build is deterministic (every SPARQL query carries an `ORDER BY`, and raw
responses are cached under `.hermes/glossary_cache/`), so re-running it on the
same Wikidata snapshot produces a byte-identical file.

## Why Wikidata and not OMW / MUSE

The brief for the broad vocabulary named NLTK's Open Multilingual WordNet (OMW)
or MUSE bilingual dictionaries. Both were checked and rejected on the evidence
on 2026-09:

### Open Multilingual WordNet (OMW 1.4, via `nltk_data`)

1. **No German.** The `omw-1.4` package contains no `deu` directory at all. The
   shipped language set is als, arb, bul, cow, dan, ell, fin, fra, heb, hrv,
   isl, ita, iwn, jpn, mcr, msa, nld, nor, pol, por, ron, slk, slv, swe, tha. A
   DE↔EN dictionary cannot be built from it — which is the one thing this
   glossary needs.

   Worth noting: `wn` (the modern OMW interface) *does* ship a German WordNet
   (odenet, CC BY-SA 4.0), but it is a **separate download** and comes with the
   licence problem below.

2. **Licence conflict.** The OMW language sets that do exist are mostly
   share-alike or attribution-encumbered:

   | Language | Project | Licence |
   |---|---|---|
   | `nld` | Open Dutch WordNet | CC BY-SA 4.0 |
   | `por` | OpenWN-PT | CC BY-SA |
   | `slv` | sloWNet | CC BY-SA 3.0 |
   | `fra` | WOLF | CeCILL-C |
   | `swe` | SALDO | CC BY 3.0 |
   | `fin` | FinnWordNet | CC BY 3.0 |
   | `ita` | MultiWordNet | CC BY 3.0 |

   klix is MIT-licensed. Bundling share-alike data inside it would impose
   CC BY-SA obligations on everyone who redistributes the package. Attribution
   licences (CC BY 3.0) are workable but still add a compliance duty to a
   package whose whole selling point is "no setup, no strings".

### MUSE bilingual dictionaries

Facebook/Meta's MUSE ground-truth dictionaries are derived from internal
translation tables and published for research use; the terms are not
unambiguously redistributable in a permissively licensed open-source package.
Rejected for the same class of reason as above.

### Chosen: Wikidata (CC0 1.0)

Wikidata is released under CC0 1.0 — a public-domain dedication with **no
attribution and no share-alike obligation**. That is the only licence class that
lets klix stay plain MIT while shipping a generated data file. Two routes are
combined, both high-precision:

**Route A — lexemes (primary).** A German *lexeme* carries a real
part-of-speech tag (`wikibase:lexicalCategory`: noun Q1084, verb Q24905,
adjective Q34698) and links through `wdt:P5137` ("item for this sense") to an
English lexeme with the same meaning. This is a genuine translation dictionary,
POS-filtered by construction. It is also free of the proper-noun noise that
Wikidata *item labels* carry — the item "Space Station" is a thing, but the
label pair for it is not vocabulary.

Measured volume at build time: **11,352** German→English lexeme pairs with a
content-word POS tag.

**Route B — curated item classes (supplementary).** German and English labels of
the same item, restricted to class subtrees whose instances are word-like
(tools, machines, materials, processes, qualities, units, foods, animals, body
parts, colours, …). This adds general vocabulary route A does not cover, while
an explicit allow-list of classes keeps people, places, organisations, works and
events out. See `CONTENT_CLASSES` in the build script.

Very deep class subtrees can exceed the Wikidata Query Service timeout (HTTP
504). That is treated as a soft failure: the class is skipped and reported, the
build continues — the lexeme route and the remaining classes already carry the
vocabulary.

## Filtering

A broad glossary is only useful if it stays precise. The build applies, in
order:

- **Content words only** — via the lexeme POS tag (route A) or the curated class
  allow-list (route B).
- **Stopword filter (DE + EN)** — the engine's own `_DEFAULT_STOPWORDS` plus an
  extended function-word list (articles, pronouns, auxiliaries, prepositions,
  discourse particles, greetings). A concept whose only German word is `und` is
  not a lookup key.
- **Minimum length ≥ 3 characters** on the ASCII-folded form — prevents acronym
  false positives (`u`, `AD`).
- **Maximum 4 tokens** — longer labels are descriptions, not words.
- **No digits, no punctuation, no URLs** — and a name-like pattern block-list
  (`university`, `gmbh`, `county`, `war`, …) as a second line of defence.
- **One surface word → exactly one concept.** A pair whose German term is
  already owned by a different concept is *dropped* and counted, not silently
  re-pointed. A glossary lookup can only return one concept, so an ambiguous
  entry is worse than a missing one. The build reports the drop count.
- **Synonym cap** — at most 4 terms per language per concept.

## Verifying the licence chain

```bash
# CC0 statement
curl -s https://www.wikidata.org/wiki/Wikidata:Licensing | grep -i -m3 "CC0"

# confirm NLTK's omw-1.4 really has no German
curl -sL https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/omw-1.4.zip \
  -o /tmp/omw.zip && unzip -l /tmp/omw.zip | grep -c "deu/"    # -> 0

# the per-language licence header the OMW tab files carry
unzip -p /tmp/omw.zip omw-1.4/nld/wn-data-nld.tab  | head -1   # CC BY SA 4.0
unzip -p /tmp/omw.zip omw-1.4/fra/wn-data-fra.tab  | head -1   # CeCILL-C
```

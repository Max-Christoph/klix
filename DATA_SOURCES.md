# Data sources and licences

klix ships four data files under `src/klix/`. Everything else in the package is
code. This file records where each comes from, the licence chain, why the
alternatives were rejected, and how the generated one is rebuilt.

## Provenance is tracked per glossary

Provenance is recorded **once per glossary**, never per concept (a format
decision, not an implementation detail). With 124 concepts you write one
`source`/`license` pair, not 124. The record lives:

* **in the file itself** for the versioned document format —
  `{"schema_version": 1, "source": ..., "license": ..., "concepts": {...}}`
  (see [`docs/glossary-format.md`](docs/glossary-format.md))
* **in `Glossary.meta`** after loading, so code can read it back
* **in `GlossaryRegistry.provenance_report()`** for several named glossaries

```python
from klix import GlossaryRegistry
reg = GlossaryRegistry(fallback="curated")
reg.register("acme", "acme_terms.json", source="Acme GmbH", license="CC-BY-4.0")
print(reg.provenance_report())
# {'acme': {'source': 'Acme GmbH', 'license': 'CC-BY-4.0', 'concepts': 42}}
```

This structure is **not hardwired to the two bundled sources.** A glossary you
add carries its own licence and shows up in the same report; nothing here needs
editing to accommodate a third or fourth source. The table below is therefore
just the *current* content of that mechanism, not its shape.

**Mixing sources inside one glossary is deliberately unsupported.** If concepts
from several origins ever need distinct licences in one file, that requires
per-term provenance — a later, separate problem. It is not pre-built.

## Shipped data

| File | Content | Origin | Licence |
|---|---|---|---|
| `data/curated_glossary.json` | **The default.** 124 concepts (manufacturing, IT, everyday office) × DE/EN, hand-written | klix author — **original work** | **MIT** |
| `data/curated_domains.json` | Domain of each curated concept, as free `tags` metadata | klix author | MIT |
| `data/glossary.schema.json` | The documented document format (JSON Schema, versioned) | klix author | MIT |
| `data/default_glossary.json` | Broad DE↔EN vocabulary, ~10k concepts, generated offline. **Opt-in** (`broad()`) | [Wikidata](https://www.wikidata.org) | **CC0 1.0** |
| `data/build_meta.json` | Build provenance of the generated file (per-class outcome, source, counts) | generated | — |
| `glossary.json` | Legacy 16-term manufacturing list, kept for compatibility | klix author | MIT |

All files are consumed read-only at runtime. Nothing is downloaded at install or
import time — `pip install klix-engine` is fully offline.

The rules behind these choices are in [`DESIGN_PRINCIPLES.md`](DESIGN_PRINCIPLES.md);
the paths that were tried and abandoned are in
[`docs/rejected-approaches.md`](docs/rejected-approaches.md).

## Why the curated file is the default and the generated one is opt-in

Measured with `evals/glossary_error_rate.py` — sense-based (an auto-resolved
concept counts as correct when its English side names the same sense as the
curated one, cosine ≥ 0.60), the curated list used as ground truth. The curated
list was written **before** the measurement, so the labels could not be shaped by
the scoring:

| Domain | curated terms tested | wrong mappings (auto) | coverage of curated terms |
|---|---|---|---|
| manufacturing | 110 | **3.5 %** | 48 % |
| IT | 94 | **8.3 %** | 26 % |
| everyday | 106 | **10.0 %** | 38 % |
| **overall** | **310** | **6.6 %** | 39 % |

And the curated set itself, checked with `evals/curated_glossary_verify.py`
(**all 124 concepts, not a sample**): 0 structural findings, 0 pairs below the
embedding-agreement floor, 0 genuine round-trip failures.

The failure mode is what decides it: a wrong mapping does not merely fail to help
— it silently bridges a query to the **wrong** English concept. 124 concepts with
no measured errors are therefore the safer default than 23,600 with a measured
6.6 % error rate. Use `broad()` when recall matters more than precision.

### Note on measurement validity

An earlier revision of this analysis reported 20 % and then 38.8 % error. Both
were wrong, and the reason matters:

* the 40-term probe was deliberately failure-enriched, so 20 % was never the
  file's error rate
* the 38.8 % figure scored by concept **key**, so `motor → engine`,
  `kryptografie → cryptography` and `kunde → client` counted as errors although
  they are correct — only the key name differed

The 6.6 % above is the corrected, sense-based figure. Separately, an
embedding-similarity metric applied to the **whole** file was discarded entirely:
it flagged `lunar eclipse / mondfinsternis` and `baptism / taufe` as weak, i.e.
it measured term rarity, not correctness. It is valid only as a pairwise
"same concept?" test between two concrete term sets, which is how it is used here.

## Rejected sources — checked at the source, not assumed

Every candidate bilingual source that would have allowed an *automatic* curated
layer is copyleft or share-alike and therefore incompatible with this MIT package:

| Source | Licence (verified how) | Verdict |
|---|---|---|
| dbnary / Wiktionary | **CC BY-SA 4.0** (Wikimedia `rightsinfo` API) | ✗ share-alike |
| FreeDict `deu-eng` 1.9 | **GPLv2+ and AGPLv3** (`COPYING` 339 lines + TEI header: *"composed of the original Ding dictionary (GPLv2+) and the ding2tei-haskell program (AGPLv3)"*) | ✗ copyleft |
| Apertium `apertium-eng-deu` | **GPL-3.0** (GitHub API) | ✗ copyleft |
| PanLex | **unverifiable** — `api.panlex.org` unreachable, HTTP 000 | ✗ unverifiable ⇒ out |
| OmegaWiki | unreachable, HTTP 000 | ✗ unverifiable ⇒ out |
| Wikidata | **CC0 1.0** — see below | ✓ **the only compatible source** |

Consequence: the multi-source **consensus principle** (a mapping is trusted when
two independent sources agree) could not be implemented. With exactly one
licence-compatible source there is no second vote, so the curated layer had to be
written rather than derived. That is the honest reason it is 124 hand-checked
concepts and not tens of thousands.

### Wikidata licence, verified

`Wikidata:Licensing` (fetched verbatim):

> "All structured data (i.e. the main, Property, Lexeme, and EntitySchema
> namespaces) is released into the public domain under Creative Commons Zero."
> "Wikidata requires a CC0 license … no other data license or designation is
> compatible with Wikidata's copyright requirements."

Cross-checked against what the builder actually reads (`scripts/build_default_glossary.py`):
`rdfs:label` and `wikibase:lemma` (main/lexeme namespaces), `wdt:P5137`,
`wdt:P31`, `wdt:P279*` (statements in the main namespace), `dct:language`,
`wikibase:lexicalCategory` (schema values) — all within the CC0 declaration.

No non-CC0 reference material is pulled in: 0 entries contain URLs, QIDs or mail
addresses, max 3 tokens per term, median 10 characters — short labels only, no
prose or reference text from third-party works.

## The generated file: what is reproducible, and what is not

* **Query-level ordering: reproducible.** Every SPARQL query carries an `ORDER BY`,
  so pagination is stable — two identical limit/offset runs return byte-identical
  rows.
* **Cache-level: reproducible.** Raw responses are cached under
  `.hermes/glossary_cache/` (keyed by a hash of the query; reads are
  corruption-tolerant, writes atomic). A build replayed from a complete cache
  yields the same file.
* **The whole artefact: NOT bit-reproducible.** Deep class subtrees intermittently
  time out on the Wikidata Query Service (HTTP 504/429). The builder logs the
  class as skipped and continues, and because it stops once the target concept
  count is reached, a run that loses one extra class can finish earlier and land
  on a **different, slightly smaller** concept set. Measured: the shipped artefact
  has **10,134** concepts; a re-run that skipped one additional deep class
  produced **10,049**. Both are valid glossaries — they are not the same file.

So treat the generated JSON as an artefact to be regenerated *and reviewed as a
diff*, not as something that can be assumed to match a previous build.
`build_meta.json` sits next to it and records the exact per-class outcome
(contributed vs. skipped), the source, the licence and the counts, so a shipped
file can be audited after the fact.

The shipped v0.9.0 file has 7 classes recorded as skipped (materials,
substances, processes, qualities, computer_terms, animals, buildings) — the
lexeme route and the other 21 classes carry the vocabulary.

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

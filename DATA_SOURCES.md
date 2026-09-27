# Data sources and licences

klix ships four data files under `src/klix/`. Everything else in the package is
code. This file records where each comes from, the licence chain, why the
alternatives were rejected, and how the generated one is rebuilt.

## Provenance is tracked per glossary

Provenance is recorded **once per glossary**, never per concept (a format
decision, not an implementation detail). With 362 concepts you write one
`source`/`license` pair, not 362. The record lives:

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
| `data/curated_glossary.json` | **The default.** 362 concepts (44 manufacturing, 186 IT, 132 everyday/support/mail) × DE/EN, hand-written | klix author — **original work** | **MIT** |
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
| IT | 369 | **11.6 %** | 19 % |
| everyday | 289 | **9.4 %** | 22 % |
| **overall** | **768** | **8.4 %** | 25 % |

This table measures the **generated** (`broad()`) vocabulary against the curated
list as ground truth. It is therefore a figure about the *generated* layer only:
the curated list is the reference here, so an error **inside** the curated list is
structurally invisible to this measurement. It is not the curated list's error
rate — for that see `docs/curated-correctness.md`. The figures moved when the
curated list grew from 124 to
362 concepts: the earlier 6.6 % was measured against 310 curated terms, the
current 8.4 % against 768. The growth added vocabulary Wikidata covers badly
(IT and everyday), so a larger, more demanding denominator raises the measured
rate. Neither number is wrong; they answer the same question over different
populations — which is exactly why the denominator is stated.

The expanded curated list itself was measured separately, on a **new random
sample** (`evals/curated_error_rate_new.py`, seed 20260928, n=160 of 362 drawn at
random — deliberately not the 310 terms above, which are now known cases):

| Domain | sampled | externally attested | ratio |
|---|---|---|---|
| manufacturing | 16 | 7 | 44 % |
| IT | 73 | 19 | 26 % |
| everyday | 71 | 20 | 28 % |
| **overall** | **160** | **46** | **28.7 %** |

*Attested* means the German term has a Wikidata lexeme whose `P5137` sense links
to an item — i.e. the concept is externally attested at all, not that the mapping
is correct. Internal ambiguity on the shipped artifact: **0**.

And the curated set itself, checked with `evals/curated_glossary_verify.py`
(**all 362 concepts, not a sample**): 0 structural findings, 0 genuine round-trip
failures. The round-trip test produces 73 misses, but each one is a **thin-anchor
artefact**, not a mapping error: the glossary bridges the German probe, the
keyword channel selects the right concept, and the dense channel outvotes it
because the anchors here are the concept's 1-3 English terms with no sentence
context. Classified individually by `evals/roundtrip_failure_diag.py`
(**73 thin-anchor, 0 mapping error, 0 sparse-also-wrong**). That script derives
its failure set from the shipped artefact; it previously carried a hard-coded
list of 10 cases, so 63 of the 73 were never classified.

The dense-agreement floor is **not** clean at this size and is not claimed as
such: 9 pairs fall below 0.35 and 30 more below 0.50, out of 362. That metric
penalises rare-but-correct terms ('lastverteiler' / 'load balancer', 0.159), so
every flagged pair is listed for manual reading with its cosine rather than
counted as an error.

### What is NOT known about the curated list

**Correctness is measured for at most 40 % of it. For the remaining 60 %, nobody
knows whether the mappings are right** — this is the honest state, not a
conservative reading of it. `evals/curated_correctness_new.py` cross-validates the
two sides against Wikidata senses (seed 20260928, n=160, same sample as the
attestation figure above):

```
CONFIRMED  (both sides share a Wikidata item) : 53
MISMATCH   (items present but disjoint)       : 11   -> 10 shown to be metric
                                                        artifacts, 1 genuine
UNDECIDABLE(no sense link on >=1 side)        : 96   -> correctness UNKNOWN
```

The undecidable remainder is concentrated in the peripheral domains, and it does
reach manufacturing:

| Domain | undecidable | of sampled | decidable so far | wrong on decidable |
|---|---|---|---|---|
| everyday | 47 | 71 | 24 | 9 (8 of them artifacts) |
| IT | 41 | 73 | 32 | 2 (both artifacts) |
| **manufacturing** | **8** | **16** | **8** | **0** |
| total | 96 | 160 | 64 | 11 |

The 8 manufacturing terms with no external ground truth are `error_code`,
`failure_cause`, `output`, `quality`, `relay`, `safety_guard`, `spare_part`,
`torque` — mostly compounded German industrial vocabulary for which Wikidata has
no lexeme sense. Manufacturing is also the **best** case on every measure that
could be decided (0 wrong of 8), so the uncertainty there is a gap in the external
source, not a suspicion about the data. A real error rate for the whole list needs
an independent labelling pass by someone other than the author; until that exists,
treat the figures above as a lower bound on the unknown, not as a clean bill of
health.

**Known polysemy risk (measured, not theorised) — a permanent property, not a bug
list.** The lookup is flat and context-free, so a German word carrying both an
industrial and an office reading resolves to whichever concept claimed it. This is
**structural**: one term maps to exactly one concept, and the format cannot express
a second assignment — `Glossary.merge` raises `GlossaryConflict` and
`validate()`/`conflicts()` report it as an ambiguous mapping. So it is not fixable
by editing the word list; that was tested, not assumed.

Scale, measured over the whole glossary (`evals/polysemy_review.py`):

```
62 of 768 curated terms (8.1 %) carry more than one Wikidata sense
  manufacturing   11   (druck, lager, prozess, verfahren, montage, mutter,
                        schraube, spannung, toleranz, ventil, wartung)
  everyday/IT     51
```

**The manufacturing exposure is contained.** All 11 polysemous manufacturing terms
resolve to a manufacturing concept; none leaks into another domain, and for 10 of
them no competing concept exists in the glossary. The single exception is `lager`:

| Query (other reading) | resolves to | should be |
|---|---|---|
| `das lager der welle ist verschlissen` (a shaft **bearing**) | `warehouse` | a part concept — **which does not exist in the glossary** |
| `der leiter ist kaputt und muss ersetzt werden` (a **ladder**) | `supervisor` | a part concept (office domain, not manufacturing) |

`lager` is therefore a **coverage gap as much as a polysemy case**: no
`bearing`/`kugellager` concept exists for the other reading to land on, so the
term can only ever mean `warehouse` here.

**Removing the synonym does not fix it.** Measured: dropping `leiter` from
`supervisor` leaves the ladder query on the office concept anyway, because the
English `supervisor` anchors are semantically close to it. The failure is the
absence of context, not the presence of a word.

Real mitigations, none of them in this release (see `docs/proposals.md`, P8):

* per-domain glossaries instead of one merged preset — the engine already supports
  it (`curated_where(tag)`, or one glossary per head)
* a context-bearing synonym: `kugellager` / `wellenlager` gives the part reading its
  own unambiguous surface form (fixes the *coverage* half of `lager`, not the bare
  `lager` case)
* context disambiguation — out of scope; the engine is deliberately stateless

Full method and adjudication: `docs/curated-correctness.md`.

The failure mode is what decides it: a wrong mapping does not merely fail to help
— it silently bridges a query to the **wrong** English concept. 362 concepts with
no measured errors are therefore the safer default than 23,600 with a measured
8.4 % error rate. Use `broad()` when recall matters more than precision.

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
written rather than derived. That is the honest reason it is 362 hand-checked
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

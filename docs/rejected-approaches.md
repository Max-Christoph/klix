# Rejected approaches

Paths that were tried, considered, or removed — with the evidence that killed
them. The point is that nobody walks the same way twice, including a future
session with no memory of the reasoning.

Entries are appended, not rewritten. If an approach becomes viable later, add a
new entry saying what changed rather than editing the old one away.

---

## NLTK / Open Multilingual WordNet (OMW) as the bilingual source

**What it was.** The obvious choice for a DE↔EN concept lexicon: a maintained
WordNet with interlingual links, installable via `nltk.download("omw-1.4")`.

**Why it looked right.** Real word sense structure, glosses, and synonyms — much
richer than bare label pairs. It is the standard answer to "where do I get a
multilingual word list".

**Why it failed — three separate reasons, each sufficient.**

1. **No German.** `omw-1.4` ships no `deu/` directory at all. The most important
   language for this project is absent.
2. **Licence.** OMW sets are predominantly CC BY-SA 4.0 / CC BY 3.0, with some
   CeCILL-C. All share-alike or copyleft, therefore incompatible with an MIT
   package (see principle 4).
3. **Semantics.** WordNet is organised around synonym sets and lexical relations;
   what is needed here is a flat `term -> concept` bridge for routing. A
   synonym-set graph would have to be flattened, discarding the structure that is
   the only reason to prefer it.

**Replaced by.** Wikidata lexemes with `P5137` senses (CC0), and — after that
proved unreliable, see below — a hand-written curated layer.

---

## dbnary / Wiktionary, FreeDict, Apertium, PanLex as bilingual sources

**What they were.** Four sources proposed specifically to get *multiple
independent* sources so a consensus rule could raise confidence: dbnary
(Wiktionary extracted to RDF), FreeDict (curated bilingual dictionaries),
Apertium (translation-oriented bilingual dictionaries), PanLex (a large
translation lexicon).

**Why they looked right.** Consensus across independent sources is the correct
way to establish confidence without hand-labelling, and these are the standard
free bilingual resources.

**Why they failed — licence, checked at the source.**

| Source | Licence | Verified how |
|---|---|---|
| dbnary / Wiktionary | CC BY-SA 4.0 | Wikimedia `rightsinfo` API |
| FreeDict `deu-eng` 1.9 | GPLv2+ and AGPLv3 | `COPYING` + TEI header of the release archive |
| Apertium `apertium-eng-deu` | GPL-3.0 | GitHub API |
| PanLex | unverifiable | `api.panlex.org` unreachable (HTTP 000) |
| OmegaWiki | unverifiable | unreachable (HTTP 000) |

All incompatible with MIT. The unverifiable ones are out by rule, not by doubt.

**Consequence, and it changed the design.** The consensus principle could not be
implemented: with exactly one licence-compatible source there is no second vote.
That is the honest reason the curated layer is 362 hand-written concepts rather
than tens of thousands of derived ones. Recorded here so it is not re-attempted
as an oversight.

**Not replaced.** Written by hand instead, and MIT-licensed as original work.

---

## Wikidata as the *sole* source for a shipped glossary

**What it was.** Generate the whole glossary from Wikidata lexemes and item
labels (CC0, so unproblematic licence-wise), at ~10k concepts.

**Why it looked right.** CC0, large, structured, and the only licence-compatible
source — so it was the rational remaining option after the licence review.

**Why it failed — measured, not guessed.** Against the curated list as ground
truth (sense-based, `evals/glossary_error_rate.py`):

| Domain | wrong mappings | coverage of curated terms |
|---|---|---|
| manufacturing | 3.5 % | 48 % |
| IT | 11.6 % | 19 % |
| everyday | 9.4 % | 22 % |
| overall | **8.4 %** | 25 % |

(Denominator: 768 curated German terms. Earlier figures — 8.3 % / 10.0 % /
6.6 % overall — were measured against a 310-term list before the curated
glossary grew from 124 to 362 concepts; see `DATA_SOURCES.md`.)

The cause is structural, not a query bug (`evals/wikidata_sense_audit.py`,
`evals/glossary_sense_check.py`):

* For `Freigabe`, `Ausfall`, `Getriebe`, `Relais`, `Drehmoment`, `Durchfluss`
  there is **no German lexeme with a linked manufacturing sense at all**.
* For `Störung`, `Ausschuss`, `Dichtung` the only linked sense belongs to a
  different domain (ecology, committee, art).
* Polysemes resolve by Wikidata's internal ordering, which is how `mutter` (the
  threaded fastener, `Q190977`, correctly linked) was mapped to `mam`.

More querying cannot invent absent data. A wrong mapping is worse than a missing
one: it silently bridges a query to the wrong concept.

**Demoted, not deleted.** Kept as the opt-in `broad()` pack for recall-oriented
use, with the error rate documented in its docstring and in `DATA_SOURCES.md`.

---

## Embedding similarity as a whole-file glossary quality metric

**What it was.** Estimate mapping correctness without hand labels by embedding
each concept's German and English sides and measuring cosine similarity — a wrong
pair should score lower than a correct one.

**Why it looked right.** It is objective, needs no labels, and scales to the
whole file. It appeared to give a usable headline number (24.3 % of pairs below
0.50).

**Why it failed.** It measures **term rarity, not correctness**. The output's
worst pairs included:

```
0.104  lunar_eclipse        de=['mondfinsternis']  en=['lunar eclipse']
0.215  sausage_casing       de=['wursthülle']      en=['sausage casing']
0.227  baptism              de=['taufe']           en=['baptism']
```

Every one of those is correct. Rare but unambiguous English/German terms simply
have low cross-lingual cosine, and a low score means "uncommon", not "wrong". The
metric cannot distinguish the two, so the number is not usable.

**Why it still exists in the repo.** `evals/glossary_mapping_quality.py` is kept
for one narrow valid use: as a **pairwise** "do these two concrete term sets name
the same concept?" test, where both sets are known. `evals/glossary_error_rate.py`
uses it in exactly that way, with a threshold, on pairs drawn from a ground-truth
list. That is a different question from the discarded whole-file estimate, and
the distinction is noted in the code.

---

## Language detection (`_guess_lang`, `_detect_lang`, `cross_lingual_only`)

**What it was.** A de/en heuristic based on signal words, used to decide whether
a query was in a different language than the anchors, with a
`cross_lingual_only` switch built on it.

**Why it looked right.** A glossary bridge is only needed for cross-language
queries, so detecting the language seems like the way to decide when to apply it.

**Why it was removed.** It was a guess, and it was unnecessary. The strictly
better guard is a **fact**: skip terms the head's own sparse vocabulary already
knows, because those need no bridge. This replaced a heuristic with a lookup and
needed no language information at all — which is also what made the glossary
language-agnostic (principle 3).

**Pitfall discovered while replacing it.** The skip set must be the *anchor*
vocabulary (`_core_vocab`), not the head's full `_vocab`. The full vocabulary
already contains the anchor-side glossary terms injected by expansion, so using
it skipped exactly the foreign-language terms the bridge needed. Effect: German
query recall went from 3/4 to 4/4 once corrected.

**Not replaced.** Deleted, per principle 3. `classifier="auto"` survives only as a
deprecated alias for `"centroid"`.

---

## `_cross_lingual_mixup` / `cross_lingual_only` as training augmentation

**What it was.** Augmentation that trained the linear probe on cross-language
anchor pairs only, intended to sharpen cross-lingual behaviour.

**Why it looked right.** It targeted the actual difficulty (cross-language
queries) instead of training on the general case.

**Why it was removed.** It was language-specific machinery in a
language-agnostic engine (principle 3), and its removal is what allows the
glossary to be the only cross-lingual mechanism.

**Honest side effect, recorded.** Removing it changed the *linear* probe's
training set from cross-language pairs only to all same-class anchor pairs (a
superset), which moved its decision boundary on exactly one of 60 frozen cases
(`SHOP` "the delivery driver never even rang the doorbell"). Measured
(`evals/centroid_regression_diag.py`):

```
code v0.8.8 (with mixup):  centroid 51   linear 51   delta  0
code v0.9.0 (without):     centroid 51   linear 52   delta -1
```

This was **not** a centroid regression — centroid never moved. It also re-framed
an older claim: centroid and linear were tied on that corpus, so
`centroid >= linear` had been passing by equality. The guard is now pinned on two
separate bounds (principle 11) rather than loosened by a blanket tolerance.

---

## Fast-path miss: comparing miss latency against a no-fast-path baseline

**What it was.** The proof that the fast-path miss overhead was eliminated: time
`decide()` with the fast path enabled on a miss, against an engine without it.

**Why it looked right.** It is the direct end-to-end measurement of the thing
being claimed.

**Why it failed.** It is not resolvable on this hardware. Two runs gave opposite
signs: −18.4 ms and +7.9 ms. The dense embedding pass (10–30 ms) dominates and
its variance exceeds the effect being measured. Both numbers were reported as
fact before this was understood — a retraction (principle 13).

**Replaced by.** A deterministic mechanism measurement
(`evals/fastpath_overhead.py`): the eliminated duplicate vectorisation costs
**~0.03 ms** median (n=300; measured 0.027–0.045 ms across runs, so only the
order of magnitude is stable), and the benchmark now counts invocations per
`decide()` — 1 on both hit and miss, versus 2 on a miss in v0.8.8. The fix is
real; its size is ~30–60× smaller than originally assumed.

**Pitfall.** The original figures also came from runs overlapped with the test
suite, i.e. a contended CPU. Interference of that kind is not visible in the
output.

---

## A blanket tolerance for `centroid >= linear`

**What it was.** Replacing the exact assertion `centroid >= linear` with
`centroid >= linear - 1` when removing `_cross_lingual_mixup` shifted one case.

**Why it seemed right.** A ±1 swing on a 60-case set is noise, and the removal was
the point of the release.

**Why it was rejected.** It cannot distinguish "centroid got worse" from "linear
got better", and it would have preserved that ambiguity permanently.

**Replaced by.** Two separately pinned bounds: `centroid >= 51` (no regression
against its own measured baseline) **and** `centroid >= linear - 1` (still at
probe level). A real regression below 50 fails loudly; a linear-only improvement
does not masquerade as a regression.

---

## Generated glossary treated as a deterministic, byte-identical artifact

**What it was.** Committing `default_glossary.json` as though a rebuild would
reproduce it exactly, and describing it as deterministic.

**Why it looked right.** Every SPARQL query carries `ORDER BY`, so pagination is
stable, and the response cache is keyed by query hash — the pipeline is
reproducible at those levels.

**Why it failed.** Running it again produced **10,049 concepts instead of
10,134**. Deep class subtrees intermittently time out on the query service; the
builder skips failed classes by design and stops once it reaches its target, so
one additional lost class lands on a different, smaller concept set.

**Replaced by.** Precise claims at each level (query ordering reproducible; cache
replay reproducible; composed artifact **not** reproducible), the file treated as
an artifact to be reviewed, and a committed `build_meta.json` recording which
classes contributed and which were skipped, so a shipped file is auditable.

---

## `{"de": [], "en": []}` as the language bucket literal

**What it was.** The build-time merge created each concept's language buckets as a
fixed pair, then appended terms into `bucket["de"]` / `bucket["en"]`.

**Why it looked right.** The curated data is DE↔EN; the buckets match the data.

**Why it failed.** It is a data fact expressed as code (principle 3): a concept
with only `{"fr": [...]}` would silently gain two empty buckets, and a
non-DE/EN source could not be represented at all.

**Replaced by.** `bucket.setdefault(lang, [])` — buckets created on demand,
whatever keys arrive. The DE↔EN restriction now lives only in the generator that
fetches DE↔EN pairs, which is allowed to be specific because that is its purpose.

---

## Domain as a validation concept, and the silent concept-key rename

**What it was.** "Domain" was a first-class notion in validation, with its own
conflict handling: a concept-key collision between domains was resolved by
renaming (`f"{domain}_{concept}"`), while a term collision raised.

**Why it looked right.** The three curated domains were real, and a
domain-qualified key keeps provenance visible in the concept name.

**Why it was removed.** Two aspects of the *same* conflict class handled
differently is inconsistent (principle 14), and the silent branch only fired for
the configuration that existed — a fourth or external glossary would have hit
neither branch. Deleting the special case was the fix; adding a second one would
not have been.

**Replaced by.** One conflict level (a term under two concepts), handled
identically for every glossary. Domains became free `tags` metadata that the
engine never reads, with `curated_where(tag)` as the filter — no list of domains
exists in the engine.

---

## An ad-hoc command as the source of a reported number

**What it was.** The figure "5.3 % of concepts are proper-name-like (533 of
10,134)" was produced by a throwaway shell command and then reported, and later
written into the proposals document as motivation.

**Why it looked right.** The command did count something real, and the examples
it printed were genuinely noise.

**Why it failed.** Three reasons, all principle 5 and 9:

* the criterion lived only in the command, so nobody could re-derive it — the
  number is not reproducible from the repository
* the criterion differed from the one used when the measurement was revisited
  (a written, narrower criterion gives 8.2 %), which is only discoverable by
  accident
* it carried no denominator statement, so "5.3 % of what, selected how" was
  unanswerable

**Replaced by.** `evals/glossary_noise.py`, which states the criterion in its
docstring, prints the two categories separately with examples, deduplicates the
overlap, and explicitly declines to report a number for taxonomic Latin rather
than report one it cannot defend.

**Lesson.** The figure had already been quoted in a proposal before the script
existed. Writing the number down is easy; writing down how it was obtained is the
part that catches this.

---

## `README` left out of a retraction

**What it was.** The retracted "1–2 ms fast-path miss penalty" was corrected in
`CHANGELOG.md` and in the benchmark's text, but not in `README.md`.

**Why it mattered.** The build backend embeds `README.md` into the wheel's
`METADATA`. The shipped artifact therefore kept advertising the number that had
just been disproven — a wrong claim in the file users actually receive.

**Corrected by.** Grepping the whole repository for the figure, including the
generated data files. This is principle 13, and it is in the checklist because
the omission was invisible until queried for.

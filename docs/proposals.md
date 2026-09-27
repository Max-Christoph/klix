# Proposals

Changes that look worth making but have **not** been built, because they were
not requested. Principle 6: scope is what was asked for; everything else lives
here until someone decides.

Each entry states the measured motivation, what it would change, the cost, and
what is uncertain. Entries are not commitments. Silence is not approval.

---

## P1 — Finish the Wikidata sense filter (raise `broad()` from 6.6 % to ~4 %)

**Motivation, measured.** `broad()` has a 6.6 % wrong-mapping rate
(`evals/glossary_error_rate.py`). `evals/wikidata_sense_check.py` shows the
cause splits cleanly in two:

* **~40 % of the wrong cases have a correct technical item linked in Wikidata
  already** — `Mutter` → `Q190977` ("Gegenstück einer Schraube, mit Innengewinde"),
  `Verschleiß` → `Q617224`, `Leistung` → `Q25342`. The builder reads `P5137` but
  ignores that a polyseme has several senses and takes whichever comes first.
  Preferring a sense whose item sits in the manufacturing/technical class would
  fix these.
* **~60 % have no usable sense at all** — `Freigabe`, `Ausfall`, `Getriebe`,
  `Relais`, `Ersatzteil`, `Taktzeit`, `Stillstand` have no P5137 link, and
  `Störung` / `Ausschuss` / `Dichtung` link only to a different domain (ecology,
  committee, art). Not fixable by querying.

**What it would change.** Maybe 6.6 % → ~4 % on `broad()`. The curated default is
unaffected — it has no measured errors.

**Cost.** Moderate: sense-selection logic in `scripts/build_default_glossary.py`,
plus a rebuild (~7 min from cache) and a re-measure.

**Uncertain.** The sense-preference heuristic needs a technical-class marker, and
that marker is itself a judgement call. Worth doing only if `broad()` gains users.

---

## P2 — Retry and re-fetch the 7 class downloads that fail intermittently

**Motivation, measured.** `src/klix/data/build_meta.json` records 7 classes that
were skipped, including `processes` (`Q1914636`) and `qualities` (`Q1207505`) —
precisely the process/property vocabulary a manufacturing glossary wants. Skipped
classes are also why a rebuild is not byte-reproducible.

**What it would change.** Better coverage in `broad()` and a more reproducible
build. Would not improve correctness (see P1) — more words, not better ones.

**Cost.** Low: smaller page sizes and a backoff retry in the builder.

**Note.** Explicitly *not* the fix for the error rate. This was one of four
"obvious" fixes proposed earlier; the measurement showed it addresses coverage,
not correctness.

---

## P3 — Place-name guard for the generated vocabulary

**Motivation, measured** (`evals/glossary_noise.py`). 834 of 10,134 concepts
(8.2 %) are name-like, in two forms:

* place/institution-qualified: 241 (2.4 %) — `altar_from_medias`, `windmill_in_auma`
* place + common noun: 593 (5.9 %) — `achel_brewery`, `abbensen_windmill`,
  `adolf_von_baeyer_gold_medal`

The existing `NAME_LIKE_RE` guard only catches German/English common nouns, so
`<place> <common noun>` slips through because the proper name comes first.

An earlier "5.3 % / 533" came from an ad-hoc command and used an undocumented
criterion; it is superseded. Taxonomic Latin is also real noise in this
vocabulary but is **not** counted — a regex cannot separate it from ordinary
two-word compounds (see the note in the script).

**What it would change.** Noise 8.2 % → ~2–3 % on `broad()`. Curated default
unaffected.

**Cost.** Low.

**Uncertain.** A stricter guard risks dropping legitimate compounds.

---

## P4 — Put the error-rate measurement in CI as a threshold

**Motivation.** The wrong-mapping rate is currently a measurement you have to
remember to run. A regression that adds ambiguous vocabulary would not be caught.

**What it would change.** `evals/glossary_error_rate.py` becomes a gate: fail if
the rate rises above a documented ceiling.

**Cost.** Low, but it needs a deterministic sampling seed and a stable
denominator — and the curated list is the ground truth, so the check only covers
terms the curated list happens to contain (39 % coverage of it).

**Uncertain.** A CI gate on a heuristic metric can produce false alarms and
invite loosening, which principle 11 warns about. Would need care.

---

## P5 — Grow the curated glossary to 300-500 concepts

**Motivation, measured.** The curated layer is the default and has no measured
errors, but it is narrow: 124 concepts, covering 39 % of the curated German terms
in the broader Wikidata vocabulary. Coverage by domain is uneven — IT 26 %,
everyday 38 %, manufacturing 48 %.

**Requested distribution** (explicitly *not* an even split): manufacturing may
stay small, since it already holds the central terms; IT and especially everyday
language should gain the most volume.

**What it would change.** Better default coverage in the two weak domains.

**Cost.** Writing time at the same quality bar — every concept through the
validator, collisions aborting, and the error rate re-measured on a **new**
random sample (not the 310 known cases, which are no longer blind).

**Status.** Approved in principle by the user; not started, because the format
work (conflict-safe merge, public API, schema) came first as the higher-value
change. Prerequisite now met.

---

## P6 — Per-term provenance

**Motivation.** Provenance is currently per glossary, so a single file cannot mix
sources with different licences. If a future glossary wants to blend curated and
generated concepts with distinct licence obligations, one header is insufficient.

**Status.** Deliberately **not** pre-built. Explicit user decision: per-glossary
provenance now, per-term provenance only if the need actually arises.

---

## P7 — Split the generated DE↔EN builder from a language-general one

**Motivation.** `scripts/build_default_glossary.py` hard-wires DE (`Q188`) and EN
(`Q1860`) in its SPARQL queries. That is intentional — building a DE↔EN
dictionary is what it is *for* — but it means the generator cannot produce, say,
a FR↔EN vocabulary.

**What it would change.** A parameterised builder, so a new language pair is
configuration rather than a second script.

**Cost.** Moderate; the query construction and the label/lemma handling both
assume the pair.

**Uncertain.** Whether a second language pair will ever be needed. The engine side
is already language-agnostic (principle 3), so this is the only remaining
language-specific component — and it is a script, not the library.

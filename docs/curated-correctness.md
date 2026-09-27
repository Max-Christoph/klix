# Correctness of the curated glossary: what can and cannot be measured

## The question this answers

The curated glossary's **error rate** — how many of its German↔English pairs
point at the wrong concept. This document exists because two earlier figures were
presented as answers to that question and neither is one:

| Figure | What it actually measures |
|---|---|
| 8.4 % / 6.6 % (`evals/glossary_error_rate.py`) | the **generated** layer's behaviour, scored against the curated English sides as ground truth. It cannot see an error *inside* the curated list, because the curated side is the reference. |
| 28.7 % "attested" (`evals/curated_error_rate_new.py`) | **coverage**: how many German terms have a Wikidata sense link at all. A term can be perfectly mapped and unattested; a term can be attested and mapped wrongly. |

## What is now measured instead

`evals/curated_correctness_new.py` — independent cross-validation. For each
sampled concept the P5137 sense items of the German side and of the English side
are looked up **separately** in Wikidata; the two sides were authored by the
glossary author, the item links were not. Same sample as the coverage figure
(seed 20260928, n=160 of 362), so the two are comparable. n=160 is small for a
percentage (the 95 % CI at 11/64 is roughly ±9 pts) — the counts are stated
alongside every rate for that reason.

```
CONFIRMED  (both sides share a Wikidata item) : 53
MISMATCH   (items present but disjoint)       : 11
UNDECIDABLE(no sense link on >=1 side)        : 96

mechanical rate on the decidable subset       : 11/64 = 17.2 %
sample that is independently decidable        : 64/160 = 40.0 %
sample with NO independent ground truth       : 96/160 = 60.0 %
```

## Adjudication of all 11 mismatches
Read individually, item labels and descriptions retrieved from Wikidata. The
question per case: **does the glossary bridge the query to the wrong concept, or
is the mismatch an artifact of how Wikidata links the English surface form?**

| Concept | German items | English items | Verdict |
|---|---|---|---|
| quotation | Q940607 *Angebot* (offer to supply) | Q206287 *Zitat* (citation), Q37113960 *Schätzwert* (estimate) | **artifact** — the German side carries the correct sense; Wikidata links the English word "quotation" to the *citation* sense. The pair is correct. |
| retention | Q759540 *Aufbewahrungsfrist* (document retention period) | Q2146123 *Retention* (medicine), Q69905337 *retention* (behaviour) | **artifact** — "retention" in the records sense is normal English; Wikidata links only the medical/behavioural senses. |
| contract_term | Q1188986 *Laufzeit* (contract term, correct) | Q1318370 *Term* (maths), Q1969448 *Benennung*, Q7702543 *term* (architecture) | **artifact** — every English item is a different sense of the surface "term". |
| supervisor | Q124291 *elektrischer Leiter*, Q1251441 *Anführer*, Q168639 *Leiter* (ladder) | Q1240788 *Vorgesetzter* (= supervisor, correct) | **genuine ambiguity (minor)** — `leiter` in German means boss *and* ladder *and* electrical conductor. The English side is right; see below. |
| deadline | Q1465133 *Frist* (time limit) | Q2404808 *Termin* (appointment) | **artifact** — adjacent concepts with separate Wikidata items; `frist`/`deadline` is a correct pair. |
| vacation | Q7124679 *Urlaub* (employee leave) | Q116213 *Ferien* (institutional holidays) | **artifact** — item granularity, not a wrong bridge. |
| appraisal | Q851176 *Beurteilung* | Q1663833 *Wertung* | **artifact** — same notion, separate items. |
| rejection | Q19358049 *Ablehnung*, Q54806642 *Absage* | Q1343202 *Verleugnung* (denial), Q98398200 *rejection* (act of declining) | **artifact** — the English side carries the correct item; Wikidata splits the German notion across two items. |
| timeout | Q578372 *Antwortzeit* (response time) | Q1771903 *Latenz* (latency) | **artifact** — near-synonyms. `antwortzeit` is a loose second synonym for `timeout`; `zeitueberschreitung` is exact. |
| return_goods | Q97167073 | Q107036510, Q1723903, Q179076, Q21651837, Q23009552, Q65088609, Q76829851 | **artifact** — the German item is a single narrowly-scoped one; the English `return`/`exchange` family is broad and polysemous. |
| datenschutz | Q456632 | Q2476929 | **artifact** — `datenschutz` / data protection: separate items, same notion. |

**Result of adjudication: 10 of 11 are method artifacts, 1 is a genuine (minor)
ambiguity.** Cleaned upper bound on the decidable subset: **≤1/64 = 1.6 %**, and
that single case is a synonym-list issue rather than a wrong concept:

* `supervisor` lists `leiter` as a German synonym. `Leiter` is genuinely
  polysemous (boss / ladder / conductor). In German text about people it reads as
  "boss", so the entry is defensible — but it is the one term in the sample where
  the flat, context-free lookup can plausibly bridge to the wrong reading.
  Recorded, not silently accepted.

## Where the undecidable 60 % sits, per domain

The uncertainty is concentrated in the peripheral domains — but it reaches
manufacturing, so it cannot be waved off as "only the office vocabulary":

| Domain | undecidable | of sampled | decidable so far | wrong on decidable |
|---|---|---|---|---|
| everyday | 47 | 71 | 24 | 9 (of which 8 artifacts) |
| IT | 41 | 73 | 32 | 2 (both artifacts) |
| **manufacturing** | **8** | **16** | **8** | **0** |
| total | 96 | 160 | 64 | 11 |

The 8 undecidable manufacturing concepts are `error_code`, `failure_cause`,
`output`, `quality`, `relay`, `safety_guard`, `spare_part`, `torque` — compounded
German industrial vocabulary for which Wikidata has no lexeme sense. Note that
manufacturing is the *best* case on every decidable measure (0 wrong of 8); the
uncertainty there is a coverage gap in Wikidata, not a suspicion of bad data.

## Known polysemy risk — measured, with reproductions

The lookup is flat and context-free: `word -> concept -> synonyms`. A German word
carrying both an industrial and an office meaning therefore resolves to whichever
concept claimed it first. Both of these were reproduced against the shipped
artifact:

| Query (meaning) | resolves to | should be |
|---|---|---|
| `das lager der welle ist verschlissen` (worn shaft **bearing**) | `warehouse` | a part/wear concept |
| `der leiter ist kaputt und muss ersetzt werden` (broken **ladder**) | `supervisor` | a part concept |

`lager` is the more serious of the two: it is core manufacturing vocabulary, and
because `lager` is claimed by `warehouse`, the *bearing* reading is pre-empted.
Similar polysemy exists in `leitung` (claim: `cable`; also management), `mutter`
(`nut`; also mother), `druck` (`pressure`; also print), `sicherung` (`fuse`; also
safeguard). None of them is a defect *of the mapping* — each is a correct sense —
but on the wrong text they bridge the query to the wrong concept.

**Removing the synonym does not fix it.** Measured: dropping `leiter` from
`supervisor` leaves `der leiter ist kaputt` on `hr_lead` anyway, because the
English `supervisor` anchors are semantically close to it. The failure is in the
absence of context, not in the vocabulary list.

Real mitigations, none of them in this release:

* per-domain glossaries instead of one merged preset (a scope change, and the
  engine already supports it via `curated_where(tag)` / one glossary per head)
* context disambiguation (out of scope — the engine is deliberately stateless)
* a context-bearing synonym: adding `kugellager` / `wellenlager` gives the bearing
  reading its own unambiguous surface form, which is a data edit but does **not**
  fix the bare `lager` case

Recorded so it is a known limitation rather than a surprise.

## The honest limit

**For 96 of 160 sampled concepts (60 %) there is no independent ground truth, and
their correctness is unknown — not assumed.** This is the real state of the
artifact after the P5 expansion, and it is a property of the *domain*, not of the
measurement: Wikidata has no German lexeme sense for compounded and office-register
terms (`ersatzteil`, `hauptplatine`, `mahnwesen`, `elternzeit`, `grussformel`), and
where it does have one, the English surface form is often polysemous enough that
cross-validation over-counts (10 of 11 cases above).

**Wikidata alone therefore cannot yield a true curation error rate.** A real figure
for the ~240 newly added concepts requires an independent labelling pass by
someone other than the glossary author — reading each pair and judging it. Until
that exists, the defensible statements are:

* internal ambiguity: **0** (mechanical, build-enforced, holds on the shipped file)
* wrong mappings among the 40 % that are externally decidable: **at most 1.6 %**,
  with 10 of 11 flagged cases shown to be artifacts
* wrong mappings among the other 60 %: **unknown**

## Reproduce

```bash
uv run python -m evals.curated_error_rate_new    # coverage: 46/160 attested
uv run python -m evals.curated_correctness_new   # this document's figures
```
Both use `random.Random(20260928)`, n=160 of 362, so they are directly comparable
and repeatable. `evals/roundtrip_failure_diag.py` classifies round-trip misses
(thin anchor vs mapping error); `evals/curated_glossary_verify.py` runs the
per-domain structural/agreement/round-trip pass over all 362.

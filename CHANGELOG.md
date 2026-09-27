# Changelog

All notable changes to klix are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versioning: SemVer.

## [0.9.1] - 2026-09-28

Documentation and measurement only. **No behaviour change, no API change, no
dependency change** — the library code is byte-identical to 0.9.0 apart from the
version string. This release closes the two known limits that v0.9.0 shipped with
and adds the script that measures the second one.

### Added
- **`evals/polysemy_review.py`** — how many distinct Wikidata senses each curated
  German term carries, over the whole glossary. Measured: **62 of 768 terms
  (8.1 %)** are polysemous, 11 of them in manufacturing. This quantifies the flat
  lookup's exposure instead of guessing at it, and it is the source of every
  polysemy figure now quoted in the README.
- **`tests/test_glossary_limits.py`** — pins both known limits so they cannot go
  stale silently. It asserts that the ambiguity is *structural* (a term cannot map
  to two concepts; `merge` refuses, `validate` reports), that the two documented
  reproductions still reproduce, and that **no part-level concept exists** for
  `lager`'s other reading. If a future change adds one, the test fails — which is
  the point, because the documentation would then be overstating the problem.

### Documented
- **The polysemy of the flat lookup is a permanent property, not a bug list.**
  One term maps to exactly one concept, so a German word with two readings resolves
  to whichever concept claimed it and the format cannot express otherwise. Scale
  and per-domain split are now in `README.md`, `DATA_SOURCES.md` and
  `docs/curated-correctness.md`.
- **`lager` (*bearing* / *stock*) is a coverage gap as much as a polysemy case.**
  It resolves to `warehouse` and no part-level concept exists for the other
  reading, so it could not resolve correctly even in principle. `leiter`
  (*ladder* / *supervisor*) is the same class but sits inside the office domain.
- **Manufacturing exposure is contained and stated as such:** all 11 polysemous
  manufacturing terms resolve to a manufacturing concept, and 10 have no competing
  concept in the glossary at all. The single exception is `lager`.
- **The 96 undecidable cases are broken down per domain** in `DATA_SOURCES.md`
  (everyday 47, IT 41, manufacturing 8), alongside what could be decided
  (manufacturing 0 wrong of 8 — the best case on every decidable measure).
- **No data was changed for any of this.** Removing the offending synonyms was
  measured and does not help: dropping `leiter` from `supervisor` leaves the ladder
  query on the office concept anyway. The failure is the absence of context, not
  the presence of a word. A real fix is a scope change — recorded as proposal
  **P8** (per-domain presets as the recommended default, plus a part-level concept),
  deliberately not implemented here.

## Unreleased (after 0.9.1)

### Added
- **`DESIGN_PRINCIPLES.md`** — 18 principles, each tied to the incident that
  produced it, plus a pre-flight checklist. Enforced by
  `tests/test_design_principles.py`, which checks that every cited file and eval
  script exists and that the quoted numbers appear in those scripts' real output.
- **`docs/rejected-approaches.md`** — abandoned paths with their disproof
  (NLTK/OMW, the four copyleft bilingual sources, Wikidata as sole source, the
  whole-file dense-similarity metric, `_guess_lang`, `_cross_lingual_mixup`,
  the wall-clock fast-path comparison, the blanket `centroid` tolerance, the
  "deterministic build" claim, the `{"de": [], "en": []}` bucket literal, domain
  as a validation concept, an ad-hoc command as a number's source, and the README
  retraction omission).
- **`docs/proposals.md`** — seven measured proposals, none built.

### Fixed
- **`evals/glossary_noise.py`** (new) replaces an ad-hoc figure: the proper-name
  share of the generated vocabulary is **8.2 % (834/10134)**, deduplicated, from a
  documented criterion. The earlier "5.3 % / 533" had no script behind it.
- The fast-path miss overhead is a **load-dependent ~0.03 ms** median (n=300;
  measured 0.027–0.045 ms across runs). Two figures had been quoted for it and
  neither survived: 0.060 ms (first measurement, taken on a contended CPU — the
  test suite was running) and 0.0264 ms (a second revision). Only the order of
  magnitude is stable, so only that is claimed. Corrected in
  `DESIGN_PRINCIPLES.md`, `docs/rejected-approaches.md`, `README.md` and the
  `glossaries.py` quality section, not just in one place (principle 13).

### Documented (known limits, after the v0.9.0 release)
- **The flat lookup's polysemy is a structural property, not a fixable entry
  list.** One term maps to exactly one concept — `Glossary.merge` raises
  `GlossaryConflict` on a second assignment and `validate()` reports an ambiguous
  mapping — so a German word with two readings can only ever resolve to the one
  that claimed it. Measured, new script `evals/polysemy_review.py`: **62 of 768
  curated terms (8.1 %) carry more than one Wikidata sense**, 11 of them in
  manufacturing. The manufacturing exposure is contained: all 11 resolve to a
  manufacturing concept, 10 have no competing concept at all. Documented in
  `README.md`, `DATA_SOURCES.md` and `docs/curated-correctness.md` — the same
  visibility as the 60 % uncertainty, not buried in an appendix.
- **`lager` is a coverage gap as much as a polysemy case.** It means *bearing* as
  well as *stock*, resolves to `warehouse`, and **no part-level concept exists**
  for the other reading, so it could not resolve correctly even in principle.
  `leiter` (*ladder* vs *supervisor*) is the same class but resolves inside the
  office domain and does not touch manufacturing.
- **No data was changed for this.** Removing the offending synonyms was measured
  and does not help: dropping `leiter` from `supervisor` leaves the ladder query on
  the office concept anyway, because the English anchors are semantically close.
  The failure is the absence of context, not the presence of a word. A fix would be
  a scope change (per-domain glossaries, context disambiguation) — recorded as
  proposal **P8**, not implemented.

### Fixed (a coverage rate was reported where an error rate was asked for)
- **The P5 measurement answers a different question than it was read as.**
  `evals/curated_error_rate_new.py` reports how many sampled German terms have a
  Wikidata sense link at all — 46/160 = 28.7 %. That is **coverage**: it says
  nothing about whether a mapping is right, and it is silent for 71.3 % of the
  sample. It was presented as the error rate. The missing measurement was then
  built.
- **A real correctness figure now exists, on an independent ground truth.**
  `evals/curated_correctness_new.py`: for each sampled concept the P5137 sense
  items of the German side and of the English side are looked up **separately**
  and compared. The author wrote the terms; the item links are external, so
  agreement is evidence rather than a restatement of the file. Same seed and same
  n=160 as the coverage figure, so the two are comparable. Result: 53 confirmed,
  11 mismatched, 96 undecidable; after reading all 11 individually, **10 are
  method artifacts and 1 is a genuine minor ambiguity**, i.e. ≤1.6 % of the
  decidable subset. `docs/curated-correctness.md` carries the adjudication table
  and the limits.
- **`evals/glossary_error_rate.py` cannot answer the question, by construction.**
  It scores the *generated* layer against the curated English sides as ground
  truth, so an error inside the curated list is structurally invisible to it. The
  6.6 %/8.4 % figures apply to the generated vocabulary only — always true, but
  the two questions had been run together. Now stated where the figure appears.
- **The honest limit, recorded rather than papered over:** for 96 of 160 sampled
  concepts (60 %) there is **no** independent ground truth, so correctness is
  unknown — not assumed. Wikidata alone cannot supply a true curation error rate,
  because it lacks senses for compounded/office-register terms and its English
  surface forms are polysemous in 10 of 11 flagged cases. A real figure for the
  ~240 new concepts needs an independent labelling pass by someone other than the
  author. That is an open item, not a solved one.

### Fixed (documentation vs. artifact drift after the curated expansion)
- **The curated counts were stale in seven places.** The expansion to 362
  concepts (44 manufacturing / 186 IT / 132 everyday) shipped, but `glossaries.py`
  still described the default as "124 concepts" and the IT and everyday presets as
  "40 concepts" each, `curated_glossary_verify.py` still printed "all 124
  concepts", and `DATA_SOURCES.md` still claimed "0 pairs below the embedding
  agreement floor". Re-measured: the verify script reports 362 concepts, and the
  floor is **not** clean — 9 pairs below 0.35, 30 more below 0.50. The claim that
  no pair fell below it is withdrawn rather than restated.
- **`evals/roundtrip_failure_diag.py` carried a hard-coded list of 10 failures**
  from the 124-concept era, so after the expansion only 10 of the (now 73)
  round-trip misses were ever classified. It now derives the failure set from the
  shipped artifact and prints a verdict per class: **73 thin-anchor artefacts,
  0 mapping errors, 0 sparse-also-wrong**. This confirms the round-trip figure
  that was previously only supported for a tenth of the cases.

## [0.9.0] - 2026-09-26

Focus: **be a fast, lightweight decision engine — not a language framework.**
Three changes, all measured, none adding a dependency.

### Fixed
- **Zero-cost fast path: the miss penalty is gone.** v0.8.8's sparse fast path
  vectorized the query inside `sparse_fastpath()` and then *again* inside
  `evaluate()` when the gate declined, so every miss paid for a second query
  vectorization on top of the dense forward pass. `decide()` now builds the
  `SparseQuery` once and hands the same object to the gate and to
  `evaluate(encoded, sparse=...)`. A miss therefore vectorizes exactly as often
  as the baseline without a fast path: once (pinned by build-count tests, not by
  wall-clock).

  Cost of the eliminated work, measured directly (`evals/fastpath_overhead.py`,
  n=300): **~0.03 ms** median per sparse-state build (load-dependent, 0.027–0.045
  ms across runs). The removed duplicate was therefore ~0.03 ms per miss — *not*
  the 1-2 ms assumed when this task was scoped. (This entry quoted 0.060 ms
  median / 0.086 ms p95 when it was written; that measurement ran on a contended
  CPU, with the test suite in parallel, and did not survive re-measurement —
  see the Unreleased section.)

  On method: a miss-vs-baseline wall-clock comparison cannot resolve this. The
  ONNX dense forward pass (~10-30 ms) dominates and its variance exceeds the
  effect, so the measured **sign flips between runs** (both −18.4 ms and +7.9 ms
  were observed for identical code on this host). The build count is the
  deterministic evidence. Fast-path hit latency is clear regardless:
  ~0.46-0.50 ms median vs a dense fallback in the tens of ms.

### Changed
- **Language-specific branches removed entirely.** `_guess_lang`,
  `_detect_lang`, `_GERMAN_SIGNAL_WORDS`, `_cross_lingual_mixup` and the
  `cross_lingual_only` expansion switch are gone. The glossary is now a flat,
  language-agnostic lookup, and its one language-dependent decision is made on
  *evidence* rather than a guess: a synonym the consumer's vocabulary already
  knows is not expanded (it bridges nothing). The skip set is the anchors' own
  tokens, not the full vocabulary — the latter already contains the anchor-side
  glossary terms and would skip the very terms a foreign query needs.
- **`classifier="auto"` is now a deprecated alias for `"centroid"`.** It used to
  guess anchor languages to choose nearest-vs-linear. Its "linear on
  single-language" half would be actively harmful now, because a broad glossary
  legitimately mixes languages inside a monolingual schema — that is not a
  reason to switch classifiers. Still parses, so old schemas keep working.
- **`translate_fn` gets its mirror language explicitly** via `translate_target=`
  (default `None` = the callback decides). No guessing from the anchor text.
- Glossary expansion is **bounded and language-balanced**: at most
  `per_concept_topk` (default 6) synonyms per matched concept — allocated
  round-robin over the concept's ISO keys, so no language crowds out another —
  and at most `max_added` overall. Round-robin fixed a real failure: with a pure
  shortest-first pick, `"urgent"` returned three English terms and one German
  one, wasting the cross-lingual bridge it exists for.
- `matched_terms` reports the cross-language terms that were actually bridged;
  the new **`matched_concepts`** field reports what the text meant in glossary
  terms (stable, and populated even when no bridge was needed).

### Added
- **`klix.glossaries` — domain packs**, so the engine carries no vocabulary:
  `empty()` (default), `manufacturing()` (the previous 16 production terms),
  `workflow()` (generic routing: error/bug, urgent, cancel, help, approve, ...),
  `default()` (the broad bundled vocabulary) and `merge_all(*packs)`.
- **`DecisionEngine(glossary=...)`** — wire one glossary for the whole schema
  instead of repeating it per head. A head constructed with its own glossary
  keeps it (explicit wins).
- **Flexible glossary input** everywhere: `Choice(glossary=...)` and
  `DecisionEngine(glossary=...)` accept a `dict`, a JSON file path, a `Glossary`
  or `None` via the new `resolve_glossary()`. A wrong type fails loudly.
- **`Glossary.validate(anchors=None)`** — structural report: duplicate terms,
  circular/ambiguous mappings (one term under two concepts), homograph conflicts
  (same word, different languages, different concepts) and collisions between
  glossary tokens and anchor text. Read-only; also exposed as
  `DecisionEngine.validate_glossary()`.
- **`Glossary.match_concepts()`**, `Choice.expand_query_terms()` and
  `SparseQuery` as documented public seams.
- **Broad bundled DE↔EN basic vocabulary** (`data/default_glossary.json`),
  generated offline by `scripts/build_default_glossary.py`. Source is
  **Wikidata (CC0 1.0)** — see `DATA_SOURCES.md` for why OMW and MUSE were
  rejected (NLTK's `omw-1.4` ships no German at all, and the OMW sets that
  exist are mostly CC BY-SA / CeCILL-C, which cannot go into an MIT package).
  Shipped: **10134 concepts / 23663 terms / 0.62 MB**, `validate()` → 0 findings.
  Size chosen from the measured curve (`evals/glossary_footprint.py`): 10k
  concepts is the largest point satisfying load, RAM and term-count budgets
  together; 13.8k overshot the load budget.
- **`data/build_meta.json`** — build provenance next to the artefact (source,
  licence, per-class contributed/skipped outcome, counts). The generated JSON is
  **NOT bit-reproducible** and is documented as such: deep class subtrees
  intermittently time out on the Wikidata Query Service, and since the builder
  stops at the target count, a run that loses one extra class lands on a
  different (slightly smaller) concept set — measured 10134 vs 10049 for two
  runs. Query ordering and the on-disk cache *are* reproducible; the assembled
  artefact is not, so it is regenerated and reviewed as a diff.

### Notes
- No new dependency. `pyproject.toml` dependencies are unchanged; the glossary
  is pure stdlib (`json`, `re`).
- `load_glossary()` (no argument) now loads the broad bundled vocabulary. The
  manufacturing pack is `klix.manufacturing_glossary()` (or
  `load_glossary(MANUFACTURING_GLOSSARY)` for the JSON copy).
- `schema_hash()` covers the glossary that actually reached a head, so an unused
  engine-level glossary does not perturb it.
- The glossary build's SPARQL cache is now corruption-tolerant (a bad entry is a
  cache miss, not a fatal error) with atomic writes. An early build died with an
  unexplained `JSONDecodeError` from a partially written entry; pinned by
  `scripts/verify_cache_robustness.py`.

## [0.8.8] - 2026-09-25

### Added
- **Weighted glossary expansion (`glossary_weight=`, default 0.5).** The
  v0.8.7 string concatenation had a structural flaw: the concatenated text is
  L2-normalized as one document, so appending synonyms *reduced* the weight of
  the text's own terms. Measured on a German anchor expanded with
  same-language synonyms, its own query match fell from 0.972 to 0.673 and the
  label flipped — that was the monolingual regression. Now the exact tokens
  keep weight 1.0 and the glossary terms enter as `alpha * v_glossary` before
  the final normalization, on the query side AND on the anchor rows.
  `alpha=0` now leaves the sparse rows bit-identical to the no-glossary
  baseline (pinned by test).

  Measured (`evals/glossary_bench.py`, alpha sweep):

  | group | baseline | alpha=0.5 | alpha=1.0 |
  |---|---|---|---|
  | EN anchors ← DE queries | 80.0 % | **100.0 %** | 100.0 % |
  | DE anchors ← EN queries | 77.8 % | **88.9 %** | 88.9 % |
  | DE anchors ← DE queries (control) | 100.0 % | **100.0 %** | 100.0 % |

  The control group is the regression case from 0.8.7 — it now stays at
  100.0 % for every alpha, i.e. **no regression**, and the cross-lingual gains
  are intact. `glossary_weight` is part of `schema_hash()`.

- **Sparse fast path (`DecisionEngine(sparse_fastpath=True)`).** Tries to answer
  from the keyword channel alone *before* paying for the dense embedding
  forward pass (~50-90 ms, the dominant cost). Three conservative gates: a
  minimum boost-weighted sparse score, a minimum margin to the runner-up
  (relative **and** absolute — the runner-up is often 0.0, which makes a purely
  relative margin degenerate to 1.0), and a reject-pole check so off-domain
  small talk never short-circuits. All-or-nothing across heads: a single head
  that cannot answer sparsely makes the whole call fall through, because mixing
  sparse and dense evidence across heads would silently change semantics.

  Measured: **0.9 ms median on a fast-path hit vs 15.4 ms on the dense path
  (~17×)**; the monolingual control group resolved 5 of 9 decisions via the
  fast path with unchanged accuracy. Hit/miss counters are exposed via
  `engine.fastpath_stats()`.

- **Glossary composition API**: `Glossary.load(path_or_dict)` and
  `glossary.merge(other)` — both return new objects, existing entries are
  extended rather than overwritten, and the merge is deterministic.

- **Explainability metadata** on every Choice result:
  `details(head)["engine"]` is `"sparse_fastpath"` or `"dense_hybrid"`, and
  `details(head)["matched_terms"]` lists the glossary terms that were active.

### Changed
- `Glossary.expand_terms(text, cross_lingual_only=False)` — the cross-lingual
  filter skips same-language synonyms. On the anchor side this avoids
  lengthening documents with terms that cannot bridge anything.
- A small stdlib-only language guess (`glossary._guess_lang`) mirrors
  `heads._detect_lang` so the glossary and the head agree on what counts as
  "the other language". It is a heuristic, not a detector; a wrong guess merely
  falls back to the previous behaviour.

### Fixed
- The v0.8.7 monolingual dilution, root-caused and pinned by tests:
  the regression came from the **anchor** side (IDF shift), not the query side
  as originally assumed. `evals/anchor_expansion_diag.py` and
  `evals/query_only_verify.py` document the four candidate designs that were
  measured before choosing the damped-anchor-row approach.

## [0.8.7] - 2026-09-25

### Added
- **`glossary=` hook for deterministic cross-lingual routing.**
  `klix.glossary.Glossary` + `load_glossary()` load a canonical → `{de, en}`
  term map from JSON. The synonyms are appended to **anchors** (in `fit()`) and
  to **queries** (in `evaluate()`), so the *sparse keyword channel* gains a
  cross-lingual bridge. Stdlib only (`json`, `re`), no model, no new
  dependency, fully deterministic.

  **Why it was needed:** the sparse vocabulary is built from the anchor texts
  alone, so a German query term (`foerderband`) could never match an English
  anchor (`conveyor`). The dense channel had to carry the whole load.
  `translate_fn` does **not** close this gap — it is only consulted on the
  `classifier="linear"` / `"hybrid"` path, never on `nearest` or `centroid`,
  which are the modes the cross-lingual benchmark runs on.

  Measured on a mixed DE/EN production-ticket set (bootstrap, 2000 resamples),
  cross-lingual groups only (19 cases):

  | classifier | baseline | with glossary | delta | 95 % CI | verdict |
  |---|---|---|---|---|---|
  | `nearest`, kb=0.5 | 78.9 % | **100.0 %** | +21.1 pt | [+5.3, +42.1] | **significant** |
  | `centroid`, kb=0.5 | 73.7 % | **94.7 %** | +21.1 pt | [+5.3, +42.1] | **significant** |
  | `centroid`, kb=1.5 | 73.7 % | **94.7 %** | +21.1 pt | [+5.3, +42.1] | **significant** |
  | `linear` | 73.7 % | 84.2 % | +10.5 pt | [+0.0, +26.3] | not distinguishable |

  Monolingual German control (DE anchors + DE queries, 9 cases): stays flat to
  ±1 case, as expected — no bridge is needed, and expansion dilutes the sparse
  vector slightly, which is the honest cost of the hook. Latency is unchanged
  (median decide 10–17 ms, dominated by the embedding pass); peak RSS 777 MB in
  the benchmark process (model weights included).

- Bundled `src/klix/glossary.json` with 16 production-domain terms (conveyor,
  cycle_time, downtime, maintenance, spare_part, shift, hydraulic, pneumatic,
  sensor, calibration, scrap, warehouse, safety_guard, error_code,
  commissioning, batch). Replace it with your own schema's vocabulary; the
  loader validates the shape and fails loudly on malformed input.
- `evals/glossary_bench.py` — the benchmark behind the table above (baseline
  vs hook, per group and aggregate, with the bootstrap CI). Cross-platform RSS
  measurement without `psutil` (POSIX `getrusage` / Windows `psapi`).
- 20 tests (`tests/test_glossary.py`).

### Fixed
- **`schema_hash()` now includes the glossary.** The hash documents itself as
  covering "every input that determines decisions"; a glossary changes them, so
  omitting it would have silently broken the reproducibility guarantee
  (log a hash, get different decisions later with the same hash). Caught by the
  dedicated test before release. The glossary state is hashed in canonical
  (sorted) form.

## [0.8.6] - 2026-09-25

### Not adopted (measured, recorded so they are not retried)
- **Squared-euclidean centroid scoring** — scoring against the class centroid
  with `-||q-c||²` instead of `cos(q, c/||c||)`, on the theory that the
  centroid's *norm* carries variance information that cosine divides away.
  Measured against the shipped cosine centroid, bootstrap 3000 resamples:

  | corpus | cosine | squared-L2 | diff | 95 % CI | verdict |
  |---|---|---|---|---|---|
  | bilingual (20) | 75.0 % | 70.0 % | −5.0 pt | [−15.0, +0.0] | not distinguishable |
  | cross-domain (70) | 84.3 % | 85.7 % | +1.4 pt | [−2.9, +5.7] | not distinguishable |
  | expanded (273) | 96.7 % | 97.1 % | +0.4 pt | [−0.7, +1.5] | not distinguishable |
  | 60 frozen | 85.0 % | 86.7 % | +1.7 pt | one case | not distinguishable |

  A control arm (squared L2 on the *unit-normalized* centroid) reproduces
  cosine **exactly** on cross and expanded — identical per-item predictions —
  which confirms the algebra and shows the norm term is the only difference.
  The hypothesis is mathematically right; the effect is simply below the noise
  floor at these anchor counts. No parameter added.
- **Calibrated linear dense+sparse fusion** (`min-max` both channels, then
  `S = α·dense + (1−α)·sparse`), as an alternative to `keyword_boost`:

  | corpus | keyword_boost | α = 0.3 / 0.4 / 0.5 | 95 % CI (α=0.3) | verdict |
  |---|---|---|---|---|
  | bilingual (20) | 75.0 % | 75.0 % | — | neutral |
  | cross-domain (70) | 71.4 % | 74.3 % | [+0.0, +7.1] | not distinguishable |
  | expanded (273) | 93.0 % | 93.8 % | [+0.0, +1.8] | not distinguishable |
  | 60 frozen | 68.3 % | 71.7 % | — | (directionally positive) |

  Every point estimate is above `keyword_boost` and no corpus regresses, but
  every CI includes zero. Notably α is **inert** — 0.3, 0.4 and 0.5 produce
  identical accuracy on all four corpora, meaning the min-max normalization
  dominates the mix and the fusion weight barely matters. Not adopted as a
  default; kept as a documented, unconfirmed mild positive.
- **Whitening — closed for good.** A second source puts the requirement at
  ~800+ unlabeled background sentences for a stable covariance estimate. We do
  not have that and will not build it; combined with the rank-deficiency
  finding from 0.8.4 (15 anchors → rank 14 covariances), the approach is
  dropped rather than deferred.
- **LEALLA backbone — not pursued.** Available only via TensorFlow Hub or a
  PyTorch conversion; using it would require a manual ONNX export through
  `transformers` + `optimum`, i.e. exactly the heavy-dependency problem that
  already disqualified `nomic-embed-text-v2-moe`. Nothing is built for it until
  the export effort is justified on its own.
- **NCMC (k-means sub-centroids, 2–3 per class)** — a real, established method,
  but only worth testing if centroid misalignment is actually *observed*
  (heterogeneous classes scoring badly). Not built preventively.
- **"ProtoKNN (Cao et al., 2026)"** — the citation looks unreliable and
  possibly misattributed. Not implemented; needs a verifiable DOI/link first.

### Added
- `evals/round2_p1.py` — the squared-euclidean experiment, runnable per corpus
  (`uv run python -m evals.round2_p1 <corpus> <mode>`), with the measured
  results in the module docstring.

## [0.8.5] - 2026-09-25

### Added
- **`classifier="centroid"`** — scores the query against the **mean anchor
  vector per label** instead of the single nearest anchor. Deterministic, no
  training, no randomness: a mean vector per class is all it computes. This
  reaches the trained linear probe's accuracy while staying in the
  "no training" contract.

  Measured on the repo's own corpora, bootstrap-verified (2000 resamples):

  | corpus | `nearest` | `centroid` | Δ | 95 % CI |
  |---|---|---|---|---|
  | 60 frozen cases | 41/60 = 68.3 % | **51/60 = 85.0 %** | +16.7 pt | — |
  | cross-domain (70) | 50/70 = 71.4 % | **59/70 = 84.3 %** | +12.9 pt | [+2.9, +22.9] **significant** |
  | expanded (273) | 254/273 = 93.0 % | **264/273 = 96.7 %** | +3.7 pt | [+1.5, +6.2] **significant** |
  | bilingual (20) | 15/20 = 75.0 % | 15/20 = 75.0 % | ±0 | neutral |

  On the 60 frozen cases it matches the trained `linear` probe exactly (85.0 %
  vs 85.0 %) — same accuracy, no training step, no model artifact. Neutral on
  the mixed-language bilingual set, so it is not a replacement for `nearest`
  there. 13 tests (`tests/test_centroid.py`), including a regression guard that
  `centroid` must not fall below the probe.
- `evals/notebooklm_proposals.py` and `evals/centroid_verify.py` — the
  experiment scripts behind the numbers above, kept for reproduction.
- `evals/metric_equivalence.py` — proves the cosine/euclidean metric question
  algebraically instead of by measurement (see Changed).

### Changed
- **RRF (Reciprocal Rank Fusion) rejected.** Replacing the weighted
  dense+sparse score addition with rank-based fusion loses heavily on every
  corpus: bilingual 75 % → 50 %, cross-domain 71.4 % → 57.1 %, expanded
  93.0 % → 89.4 % (k=40 and k=60 both). Recorded so it is not retried.
- **Metric switch cosine → euclidean: not done, and no test needed.** For
  L2-normalized vectors `||a-b|| = sqrt(2-2·cos)` exactly (verified: max
  deviation 4.4e-16, 0 ranking mismatches in 2000 trials), so the ranking is
  provably identical. Beyond that, `keyword_boost` and every calibrated
  threshold (`reject_threshold`, `min_coverage`, the Flag softmax) live on the
  cosine *scale*, which euclidean is not comparable to — a switch would need
  full recalibration for zero ranking benefit.
- **`pytest` now puts the repo root on `sys.path`** (`pythonpath = ["."]`) so
  tests can import the `evals/` harness, which is not part of the shipped
  package.

## [0.8.4] - 2026-09-25

### Added
- `DecisionEngine(truncate_dim=N)` — opt-in embedding truncation (MRL-style:
  slice + L2 renorm, no training, fully deterministic). Wired through
  `HybridBackbone`, which wraps `embed_model.embed` so anchors and queries are
  transformed identically. Verified: default `None` is unchanged behaviour;
  `truncate_dim` reproduces the measured numbers exactly. 9 dedicated tests
  (`tests/test_truncate_dim.py`).
- `evals/backbone_compare.py` — backbone and embedding-transform shootout on the
  repo's own corpora (60 frozen cases, 70-case harness, 273-case expanded
  corpus). Answers three questions empirically instead of trusting MTEB
  leaderboard numbers; the headline results are in the module docstring.

### Measured (documented, not defaulted)
- **Backbone swap rejected.** Granite-Embedding-97M-Multilingual-R2 was
  evaluated as a drop-in replacement (registerable via FastEmbed's
  `add_custom_model`, 98 MB quantized — less than half of MiniLM). Result on
  the 60 frozen cases: nearest 42/60 = 70.0 % vs MiniLM 41/60 = 68.3 %
  (+1.7 pt, inside the ±9 pt CI at n=70 → noise); linear 47/60 = 78.3 % vs
  85.0 % (−6.7 pt, consistent across every domain). The loss is **not** a
  hyperparameter artifact: sweeping `classifier_C` from 0.5 to 200 keeps
  Granite at 78–80 % while MiniLM stays at 85 %. Its better MTEB *retrieval*
  score does not transfer to few-anchor cosine classification.
- **Whitening does not work at this anchor scale — and the reason is
  structural.** With ~15 anchors in 384 dims the anchor covariance has rank 14
  (370 dims carry zero anchor variance). ZCA therefore collapses all pairwise
  cosine similarities to a single value (off-diagonal std = 0.0000), so
  `nearest` degenerates to tie-breaking (accuracy unchanged at every dims
  setting), and the `linear` probe drops from 96.7 % to 40.3 % because the
  zero-variance dims are amplified 1000× as pure query noise. ZCA needs
  `n_anchors >> dim` (rule of thumb ≥10×); here it is 15 vs 384. This is a
  limit of the method at few-shot scale, not an implementation bug — hence no
  `whitening=` parameter is added.
- **Truncation (`truncate_dim`) is real but small, and kept opt-in.** Slicing
  MiniLM embeddings + re-normalizing (pure slice + L2 renorm, deterministic)
  improves accuracy consistently on all three corpora: 60 cases 68.3 % → 76.7 %,
  70 cases 71.4 % → 77.1 %, 273 cases 93.0 % → 94.9 % (at 64 dims). The effect
  is **not monotone** on the frozen sets (96 dims dips below 128 dims), so it is
  documented as an opt-in knob rather than a new default — and it lowers cosine
  cost proportionally, which is why it is worth having at all.
- **Not testable / not practical (recorded so nobody re-researches it):**
  `embeddinggemma-300m` is a gated model whose ONNX export needs a separate
  309 MB external-data file that `model_file` cannot fetch;
  `nomic-embed-text-v2-moe` is absent from FastEmbed (only v1/v1.5 ship), its
  one ONNX repo holds no model files, and it *requires* task prefixes;
  `Qwen3-Embedding-0.6B` needs a 613 MB quantized ONNX, `last_token` pooling
  (FastEmbed offers CLS/MEAN/DISABLED only) and instruction prefixes.

## [0.8.3] - 2026-09-24

### Documentation
- **README restructured for a first-time reader.** The "Why klix — and when it
  isn't the right tool" section (honest framing, "you could write the same in
  30 lines", "a trained classifier will beat it") sat in the first 40 lines,
  before the reader knew what the library did. Content is unchanged and
  complete — only the order moved:
  - **Now first:** one sentence stating what it does ("sort text into
    categories, get yes/no flags, score on an axis — by writing example
    sentences instead of training a model"), an 11-line runnable snippet with
    its actual output, and the no-GPU/no-labels/offline property.
  - **Moved down:** the honest framing, now *after* "What you get" and
    "Quickstart", introduced as a single paragraph that links to the full
    section (placed just before the Benchmarks, where the limits belong).
  - **New, in the honest section:** a pointer that not every question is a
    text question — urgency / business impact / SLA risk / customer tier are
    decided by context the message does not carry, with a reference to the
    `Score` docstring's measured evidence. This is the one design trap worth
    knowing before writing a schema.
- Both README code examples were executed and verified to produce exactly the
  output the comments claim (`res.queue == 'ot_plant'`), since the first
  snippet is the one thing a visitor may actually run.

## [0.8.2] - 2026-09-24

### Fixed
- **`translate_fn` failures are no longer silent.** A `translate_fn` that
  raised was swallowed by a bare `except Exception: pass`; the schema was
  then quietly weaker (no cross-lingual anchors) with no hint why. It still
  never breaks `compile()` — best-effort remains the contract — but now
  reports once per compile via `UserWarning`, naming the error type and how
  many anchors went unmirrored. 7 new tests pin the behaviour, including the
  counter-case (a translator returning `None` is not an error).
- **`translate_fn` scope documented.** It only augments the
  `classifier="linear"` / `"hybrid"` training matrix; on the `nearest` path
  (the default) the hook is never called. Previously undocumented, so users
  on `nearest` could reasonably expect cross-lingual mirroring that never
  happened. Now stated in the README parameter table, in `Choice.__init__`,
  and pinned by a regression test.

### Documentation
- **Benchmark tables now state the anchor count.** The cross-domain numbers
  (72 % nearest / 84 % linear) were measured with **3 anchors per class**;
  reading them without that context invites the wrong conclusion ("weak
  engine" instead of "small anchor set"). A new table shows the same engine
  at 3 / 9 anchors and with/without rules (57 % → 87 % → 93 % → 97 %), and
  states plainly that anchor coverage is the dominant quality lever.
- **`Score` docstring documents when NOT to use the head.** Four measured
  mechanisms against a context-dependent target (urgency) all returned the
  same number (9/20, 16→15/20, 11/20, 9/20) — the ceiling is structural.
  The docstring now names the property classes that genuinely are text
  properties (tone, specificity, scope) versus those decided by context the
  message does not carry (urgency, business impact, SLA risk, compliance
  exposure, customer tier), and gives the cheap self-check plus the correct
  architecture (head supplies inputs, application does the valuation).

## [0.8.1] - 2026-09-24

### Changed
- **Honest positioning in the README**: new "Why klix — and when it isn't
  the right tool" section. Klix is explicitly a packaging/product decision
  (declarative reviewable schema, honest uncertainty, explainability, no ML
  infra), not an algorithm innovation; default-mode accuracy is on par with
  a trivial dense Embed-KNN (76 % vs 78 %), the accuracy edge appears only
  in specific modes (`linear` on single-language schemas: 86 %). The
  trade-off vs. writing ~30 lines of sentence-transformers + sklearn is
  stated outright, together with the use cases where klix helps and where
  a trained classifier or a copy-pasted snippet is the better choice.
- **Contradiction fixed**: the "Choosing a variant" section no longer lists
  hard-negative mining as a "+7 pts measured" accuracy lever. The measured
  holdout result is ≈0; the +7 pts came from memorization of the mining
  set and is now explicitly marked as corrected in 0.8.1.
- **Timing tests moved out of the CI gate**: the two wall-clock assertions
  (head-evaluation latency, batch-vs-serial throughput) are marked
  `@pytest.mark.benchmark`; `pytest` defaults to `-m "not benchmark"` and
  CI runs the functional gate only. Timing assertions under shared CPU load
  are flaky by nature and previously masked a real failure (0.8.0 report).
  Benchmark tests still run locally via `uv run pytest -m benchmark`.
- **Latency claims corrected across the repo**: the "< 1 ms for 3 heads"
  and "~10 ms forward pass" figures are superseded by measurement
  (2026-09-24): head evaluation for the fixture schema is ~1.4 ms median
  (budget raised to 5 ms — the old 1 ms bound predates both the larger
  schemas and FastEmbed 0.8 mean pooling, and had started failing for real,
  not flakily), and the embedding forward pass measures ~50-90 ms on the
  dev workstation. Updated in README, `src/klix/engine.py`,
  `src/klix/heads.py`, `examples/production_pattern.py`, and the benchmark
  script's own latency note.
- **`evals/` catalog added to the README**: a table distinguishing live
  results (benchmark.py, benchmark_bilingual.py, setfit_baseline.py,
  hard_negative_e2e.py, eval_domains.py) from historical experiments.
- **Benchmark numbers re-measured and corrected**: the FastEmbed release
  in the lockfile (0.8.x) computes mean-pooled MiniLM embeddings instead
  of the previous CLS pooling. On the fixed 60-case 5-domain set the klix
  accuracy is unchanged (nearest 41/60 = 68 %, linear 51/60 = 85 %,
  verified via `evals/linear_sweep.py`); the cross-domain harness
  (`benchmark.py`, incl. GUARD set) moved `klix nearest` 76 % → 72 % and
  `klix linear` 86 % → 84 %, and the bilingual EN/DE split shifted
  (nearest 9/6 → 8/7, topk2 9/6 → 9/7, linear 10/5 → 9/6). The README
  tables carry a version note explaining the change; latency claims
  (~10 ms forward pass) are unchanged.
- **`evals/` execution convention unified**: scripts are run as modules
  from the repo root (`uv run python -m evals.<name>`) so package imports
  (`from evals.X import ...`) work consistently; previously some scripts
  used flat imports (`from linear_sweep import ...`) that silently broke
  when run from the repo root. README and CI updated to the module form.

### Fixed
- `examples/production_pattern.py` section in the README was in German in
  an otherwise English document — translated.

## [0.8.0] - 2026-09-24

### Added
- `classifier="hybrid"` (opt-in): logistic probe learns the dense/sparse
  weighting on concatenated `[dense | tfidf]` features instead of the
  hand-tuned `keyword_boost`. Wins on keyword-rich schemas (asset IDs, SKU
  codes); on the 60-case benchmark it trails dense-only `linear` (81.7% vs
  85.0%) — keep `nearest`/`linear` as defaults.
- `sparse_metric="bm25"` (opt-in): BM25 replaces the TF-IDF cosine channel
  (length-normalized, tf-saturated). 43/60 vs 41/60 on the labeled sets;
  combined with `topk3 + coverage` it reaches 85% without any probe training.
- `bm25_k1` / `bm25_b` parameters (defaults 1.5 / 0.75, sweep-validated).
- Hard-negative mining workflow (`HardNegativeStore`, `attach_counterexamples`,
  `Choice.add_counterexamples`): collect low-confidence live decisions, review
  them (human-in-the-loop by design — no auto-training), attach them as
  label-specific counterexamples, recompile. **Holdout-verified (v3 eval):
  generalization gain on unseen cases ≈ 0** (11/20 → 10/20, n=20); the
  earlier 68.3% → 75.0% was memorization of the mining set. Repositioned as
  a diagnosis loop (surfaces confused label pairs), not an accuracy lever.
- `DriftMonitor` (E.2): streaming monitor over Score coverage / Flag margin /
  Choice confidence with sliding-window low-trust-share alerting (cooldown
  prevents alert storms). Never raises into inference.
- `DecisionEngine.schema_hash()` (E.3): stable SHA-256 over the full schema
  configuration (anchors, counterexamples, rules, calibrated thresholds,
  stop words, model config) — the schema hash IS the model version for
  reproducible historical decisions.
- SetFit baseline comparison (E.4, `evals/setfit_baseline.py`): honest
  benchmark vs. trained few-shot classification on the same examples.
  Leakage-verified (anchors and test cases are disjoint — zero exact/near
  duplicates). Result: SetFit matches the linear probe (51/60 = 85%) and
  buys nothing beyond it; klix's pitch is equivalent accuracy with instant
  schema updates + explainability. Documented in the README.
- Bootstrap 95% confidence intervals in `calibrate()` reports; CI lower bound
  is floored for tiny samples (n < 8) so a perfect point estimate cannot
  masquerade as certainty.
- `LLM-guardrail / pre-routing` eval scenario (faq/llm/human gatekeeper).
- `CHANGELOG.md`, `CONTRIBUTING.md`, CI badge; README: statistical-honesty
  disclaimers on n=70 tables, offline/air-gapped deployment guide,
  release chain updated to Trusted Publishing (OIDC).

### Fixed
- Legacy v0.7.x hard-gate contract tests now pin `soft_coverage=False`
  explicitly (they broke under the new soft-coverage default).
- Dead code removed from the BM25 query path.

## [0.7.2] - 2026-09-23

### Added
- `Score.fallback_value`: off-axis texts below `min_coverage` return this value
  instead of `None` (for None-intolerant pipelines like Jira/Salesforce).
- `validate_anchors()` suggestions now include rewrite/merge advice.
- Honest batch benchmarks in docstring + README; `bench_batch.py` eval.

## [0.7.1] - 2026-09-23

### Fixed
- `Rule.compile()` raises `ValueError` naming the rule on invalid patterns.

## [0.7.0] - 2026-09-23

### Added
- Enterprise hardening: `Score.min_coverage` gate (noise -> `None` + `raw_value`),
  loud calibration `UserWarning` when n < 20, politeness fillers filtered.

## [0.6.1] - 2026-09-23

### Docs
- Rules docstring (boost does not resurrect rejected results), bug-hunt script.

## [0.6.0] - 2026-09-23

### Added
- `validate_anchors()`: read-only class-overlap report with confuser terms,
  sharpening hints, misplaced anchors.

## [0.5.0] - 2026-09-23

### Added
- Robustness wave: extended stopwords, CV-based `calibrate()`, `decide_batch`.

## [0.4.0] - 2026-09-23

### Added
- Declarative rules (force/boost), `calibrate()` for Flag/Score/Choice,
  `explain()` decision attribution.

## [0.3.0] - 2026-09-22

### Added
- `classifier="auto"`, `translate_fn` cross-lingual augmentation, bilingual
  stopwords, honest Flag calibration (margin/coverage), CI benchmark job.

## [0.2.1] - 2026-09-22

### Added
- `Flag(aggregation="topk")`; LLM-guardrail eval + tests; char-n-gram
  experiment (documented negative result).

## [0.2.0] - 2026-09-22

### Added
- `Choice classifier="linear"` (logistic probe + anchor augmentation):
  decision boundary instead of nearest-anchor distance. 41/60 -> 51/60 on the
  labeled sets.

## [0.1.7] - 2026-09-22

### Fixed
- topk per-label clamp, coverage boost math, reject-pole hybrid scale,
  `reject_threshold`.

## [0.1.6] - 2026-09-22

### Fixed
- Per-head TF-IDF vectorizers (true decoupling — fixes shared-IDF leak),
  fast in-house query vectorizer.

## [0.1.5] - 2026-09-22

### Added
- `DecisionEngine(stop_words=...)`; honest README (config table).

## [0.1.4] - 2026-09-22

### Added
- `Choice.reject_anchors`, `Score` topk aggregation + coverage signal,
  eval harness.

## [0.1.3] - 2026-09-22

### Changed
- Fully internationalized package (English docs, metadata); corrected author.

## [0.1.2] - 2026-09-22

### CI/CD
- Publish workflow end-to-end test via tag push.

## [0.1.1] - 2026-09-22

### Added
- `project.urls`, README badges.

## [0.1.0] - 2026-09-21

### Added
- Initial release: shared-backbone engine with decoupled decision heads
  (Choice, Score, Flag), FastEmbed MiniLM + per-head TF-IDF, rules, demo.
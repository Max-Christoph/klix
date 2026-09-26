# Changelog

All notable changes to klix are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versioning: SemVer.

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
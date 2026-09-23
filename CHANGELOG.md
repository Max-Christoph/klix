# Changelog

All notable changes to klix are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versioning: SemVer.

## [Unreleased]

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
  label-specific counterexamples, recompile. End-to-end verified: 41/60 (68.3%)
  -> 45/60 (75.0%) on the labeled sets.
- `DriftMonitor` (E.2): streaming monitor over Score coverage / Flag margin /
  Choice confidence with sliding-window low-trust-share alerting (cooldown
  prevents alert storms). Never raises into inference.
- `DecisionEngine.schema_hash()` (E.3): stable SHA-256 over the full schema
  configuration (anchors, counterexamples, rules, calibrated thresholds,
  stop words, model config) — the schema hash IS the model version for
  reproducible historical decisions.
- SetFit baseline comparison (E.4, `evals/setfit_baseline.py`): honest
  benchmark vs. trained few-shot classification on the same examples.
  Result: SetFit matches the linear probe (51/60 = 85%); klix's pitch is
  instant schema updates + explainability, not "training accuracy at zero
  cost". Documented in the README.
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
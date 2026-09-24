# Changelog

All notable changes to klix are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versioning: SemVer.

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
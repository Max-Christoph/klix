# Klix

[![PyPI version](https://img.shields.io/pypi/v/klix-engine.svg)](https://pypi.org/project/klix-engine/)
[![Python](https://img.shields.io/pypi/pyversions/klix-engine.svg)](https://pypi.org/project/klix-engine/)
[![CI](https://github.com/Max-Christoph/klix/actions/workflows/ci.yml/badge.svg)](https://github.com/Max-Christoph/klix/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Decoupled decision heads on a shared semantic backbone.
A text passes through the embedding model **exactly once** (dense + sparse), after which any number
of heads (`Choice`, `Score`, `Flag`) operate on the precomputed vectors — each in its own
mathematical space. No model training, no slot limits, fully offline and CPU-only.

## Why klix — and when it isn't the right tool

Klix is a **packaging** decision, not an algorithm decision. The core
(embedding + KNN / logistic probe) is standard practice since
sentence-transformers popularized it in 2019; you could write an equivalent
~30-line snippet with `sentence-transformers` + `sklearn`. What klix adds is
the product around that core:

- a **declarative, reviewable schema** (`Choice` / `Score` / `Flag` heads)
  that lives in the repo, doubles as documentation, and updates in
  milliseconds — no training step, no model artifact per schema;
- **honest uncertainty handling** out of the box (reject poles, `coverage`
  signals, calibrated thresholds) instead of a forced guess;
- **explainability** (token-level attribution) and **no ML infrastructure**:
  no GPU, no API keys, fully offline after a one-time model cache.

**Where it genuinely helps:** rapid prototyping of text-routing logic
without labeled data and without an ML pipeline — e.g. coarsely sorting
incoming tickets or fault reports into categories when you have neither the
time nor the data volume for a trained model, and plain keyword matching is
too brittle. Small volumes, clearly separable categories, no
training-data-pipeline required.

**Where it does not:** with a few hundred labeled examples per class, a
trained classifier will beat it; and for a single throwaway routing problem,
copy-pasting the 30-line KNN/LogReg snippet is simpler than adopting a
library. Measured default mode is on par with a trivial dense Embed-KNN
(72 % vs 78 % in the cross-domain benchmark) — the accuracy edge appears
only in specific modes (`classifier="linear"` on single-language schemas:
84 %). See the [Benchmarks](#benchmarks) section for the honest numbers.

## Architecture

```text
Text ──► HybridBackbone (FastEmbed dense + TF-IDF sparse, once, tens of ms)
              │
              ├──► Choice   (routing/classification: max-similarity + keyword boost)
              ├──► Score    (continuous axis: low/high anchors + sigmoid)
              ├──► Flag     (boolean: 2/3-class softmax with temperature)
              └──► custom heads (subclass BaseHead)
```

- **Shared backbone:** Each text is embedded and TF-IDF-transformed exactly once.
  Whether you register 3 heads or 50, the extraction cost stays the same.
- **Decoupled heads:** Adding options to one `Choice` never affects `Score` or `Flag`
  results. Every head encapsulates its own logic.
- **Declarative:** Define schemas with example sentences, call `compile()`, done.
- **Honest uncertainty:** Optional reject poles (`Choice(reject_anchors=...)`,
  `Flag(neutral_anchors=...)`) return `None` instead of guessing; the `Score` head
  reports a `coverage` signal so you know when a score is noise.
- **Language-agnostic anchors:** The backbone model is multilingual, so anchor
  sentences in any language work — German, English, mixed, whatever fits your domain.

## Installation

```bash
uv add klix-engine
# or
pip install klix-engine
```

On first use, FastEmbed downloads the `paraphrase-multilingual-MiniLM-L12-v2` model
(~120 MB, one-time, then cached locally). Everything runs offline afterwards.

## Quickstart

```python
from klix import DecisionEngine, Choice, Score, Flag

engine = DecisionEngine()

engine.add_head(
    Choice(
        name="target",
        options={
            "it_ops": ["VPN down", "server unreachable", "laptop won't boot"],
            "ot_plant": ["robot cell stopped", "PLC fault", "plc-34 error", "cycle time deviation"],
            "finance": ["cost center over budget", "approve invoice"],
            "facility": ["oil spill in hall 2", "heating broken"],
        },
        # Optional: texts resembling these get value=None instead of a forced guess.
        reject_anchors=["casual office chat", "birthday wishes", "off topic request"],
    )
)

engine.add_head(
    Score(
        name="urgency",
        low_anchors=["routine maintenance", "casual question"],
        high_anchors=["emergency right now", "production line down", "acute danger"],
        min_val=0.0,
        max_val=3.0,
        # "topk" pools the best 2 anchors per pole (robust against a single
        # noisy anchor). Every result carries "coverage": if it is low (< ~0.3),
        # the text matched neither pole and the score is mostly noise.
        aggregation="topk",
    )
)

engine.add_head(
    Flag(
        name="is_security",
        true_anchors=["hacker attack", "ransomware infection", "data exfiltration"],
        false_anchors=["hardware broken", "ordinary IT problem", "network outage"],
        # Optional third pole: when "neutral" wins, value=None instead of True/False.
        neutral_anchors=["routine request", "general question", "other topic"],
        threshold=0.5,
    )
)

engine.compile()

res = engine.decide("plc-34 reports a fault, conveyor belt stopped immediately!")

print(res)                                   # e.g. <DecisionResult (61 ms): target=ot_plant, urgency=2.4, is_security=False>
print(res.target)                            # 'ot_plant'
print(res.urgency)                           # continuous score between 0.0 and 3.0
print(res.is_security)                       # True / False / None (neutral won)
print(res.details("is_security"))            # full dict: value, probability, probabilities
```

Exact score values depend on your anchors — always treat the outputs as calibrated
signals, not ground truth, and tune the anchors to your domain.

## Configuration

Every head and the engine expose meaningful knobs:

| Knob | Where | Effect |
|------|-------|--------|
| `options`, anchors | all heads | The schema itself — more/better example sentences are the main quality lever |
| `classifier` | `Choice` | `"nearest"` (default), `"linear"`, or `"auto"`. `"auto"` picks `"nearest"` for mixed-language anchors (robust) and `"linear"` for single-language (highest accuracy) |
| `classifier_C` | `Choice` | Regularization strength for the linear probe (lower = more regularization, use with few anchors) |
| `translate_fn` | `Choice` | Optional `(text, target_lang) -> str` hook: mirrors each anchor into the missing language at compile time, closing the cross-lingual gap without writing anchors twice. **Only active on the `classifier="linear"` / `"hybrid"` path** — it augments the probe's training matrix, which `nearest` does not have; on `nearest` the hook is silently unused. A `translate_fn` that raises is reported once per compile via `UserWarning` (it never breaks `compile()`) |
| `reject_anchors` | `Choice` | Texts matching these return `value=None` (don't-know instead of guess); with `classifier="linear"` they are learned as their own class |
| `keyword_boost` | `Choice` | Weight of exact keyword hits (asset IDs like `plc-34`) vs. semantic similarity |
| `aggregation` | `Score` | `"max"` (default) or `"topk"` — topk averages the best-k anchors per pole, robust against a single noisy anchor |
| `coverage` | `Score` result | Pooled similarity to the better pole; low (< ~0.3) means the score is noise |
| `min_val` / `max_val` / `sharpness` | `Score` | Output range and sigmoid steepness |
| `neutral_anchors` | `Flag` | Third pole for out-of-domain: returns `value=None` when it wins |
| `aggregation` | `Flag` | `"max"` (default) or `"topk"` — topk averages the best-k anchors per pole, robust against a single noisy anchor |
| `threshold` / `temp` | `Flag` | Decision cutoff and softmax temperature (lower = sharper) |
| `model_name` | `DecisionEngine` | Any FastEmbed-compatible embedding model |
| `stop_words` | `DecisionEngine` | Custom stopword list for the TF-IDF index (default: extended EN+DE list filtering grammatical fillers; pass `[]` to disable filtering) |
| `evaluate(encoded)` | `BaseHead` subclass | Add entirely custom head types (regex, business rules, ...) |
| `rules` | `Choice` | Hard keyword/regex `Rule`s (force/boost) layered over the semantic decision |
| `engine.calibrate(head, samples)` | `DecisionEngine` | Learn Flag threshold / Score sharpness+remap / Choice reject_threshold from labeled samples; k-fold CV for n ≥ 6, stability reported via `spread` |
| `res.explain(head)` | `DecisionResult` | Token-level attribution: which keywords and which anchor drove the decision |
| `engine.decide_batch(texts)` | `DecisionEngine` | Bulk mode: one embedding pass for the whole list — per-item overhead drops sharply for large volumes |
| `engine.validate_anchors()` | `DecisionEngine` | Read-only anchor-quality report: overlapping classes (centroid cosine), shared confuser terms, sharpening hints, misplaced and duplicate anchors. `validate_anchors_report()` returns a formatted string |

## Custom Heads

Subclass `BaseHead` and implement `evaluate(encoded)`:

```python
import re
from klix import BaseHead

class RegExExtractionHead(BaseHead):
    def __init__(self, name: str, pattern: str):
        super().__init__(name)
        self.re = re.compile(pattern)

    def get_reference_texts(self) -> list[str]:
        return []  # no reference texts needed

    def fit(self, backbone) -> None:
        pass

    def evaluate(self, encoded) -> dict:
        match = self.re.search(encoded.text)
        return {"value": match.group(0) if match else None}
```

## Production Pattern

`examples/production_pattern.py` shows the complete production-ready pattern in a
single file: Choice/Score/Flag with `classifier="auto"` and `translate_fn`, an explicit
catch-all class (`not_relevant`), the two-stage action pattern (security flag as
veto, confidence gate, auto-close), a custom head via `BaseHead` subclassing,
and the interpretation of all confidence signals.

```bash
uv run python examples/production_pattern.py
```

## Benchmarks

All benchmarks are **reproducible** — scripts live in `evals/` and every method is
trained/evaluated on the *same* labeled data (the anchors are the few-shot training set).

### Bilingual routing (EN/DE, 5 classes, support tickets)

`evals/benchmark_bilingual.py` — 10 English + 10 German test cases with parallel
meaning, anchors mixed EN/DE. Reproduce with:

```bash
uv run python -m evals.benchmark_bilingual
```

| Method | EN | DE | Combined |
|---|---|---|---|
| TF-IDF + LogReg | 8/10 | 6/10 | 14/20 |
| Embed-KNN (dense) | 9/10 | 7/10 | **16/20** |
| klix nearest | 8/10 | 7/10 | 15/20 |
| klix nearest + topk2 | 9/10 | 7/10 | 16/20 |
| klix linear | 9/10 | 6/10 | 15/20 |

**Latency per decision (CPU, includes the embedding forward pass):**
TF-IDF+LogReg ≈ 1–5 ms (no embeddings); embedding-based methods
≈ 60–80 ms, dominated by the ~50–90 ms MiniLM forward pass. Measured
2026-09-24 on the dev workstation; absolute values are
hardware-dependent (see the version note below).

**Reading this honestly:** on this *mixed-language, few-anchor* schema the
`linear` probe overfits to English (90 % EN / 60 % DE). The dense Embed-KNN is
the most language-robust. Recommendation: with few mixed-language anchors, use
`classifier="nearest"`; the `linear` probe pays off on *single-language* schemas
with several anchors per class (see the cross-domain result below).

### Cross-domain routing (6 domains, 70 cases, mostly EN)

`evals/benchmark.py` — HR, Finance, Image-captions, Tasks, Shop, and the
LLM-guardrail scenario (n=70; note: five of the six domains are the same
labeled sets used in the SetFit comparison below, plus the GUARD set):

| Method | avg accuracy | median latency |
|---|---|---|
| TF-IDF + LogReg | 51 % | ~1–5 ms |
| Embed-KNN (dense) | 78 % | ~58 ms |
| klix nearest | 72 % | ~71 ms |
| **klix linear** | **84 %** | ~80 ms |

Here `klix linear` is the accuracy winner (+12 pts over the nearest-anchor
ceiling). Latency is dominated by the embedding forward pass and scales
with hardware (measured on the dev workstation, 2026-09-24).

**Read the anchor count with these numbers — they are not comparable
without it.** This benchmark uses **3 anchors per class** (the eval sets
were kept deliberately small so the comparison against TF-IDF/Embed-KNN is
honest on identical data). With 3 anchors per class, 72–84 % is the
expected range, not a ceiling. The dominant quality lever is anchor
*coverage of the class*, not the algorithm:

| Anchors per class (6 classes, same engine) | Category accuracy |
|---|---|
| 3 — instance-style ("plc-34 meldet fehler") | 57 % |
| 3 — phenomenon-style ("sps fehlermeldung an der anlage") | 87 % |
| 9 — phenomenon-style, no rules | 93 % |
| 9 — phenomenon-style + 2 `Rule`s for asset tags | **97 %** |

So a reader who sees "3 anchors → 72 %" and concludes "weak framework" is
measuring the anchor set, not the engine. Two separate levers, both free:
phrase anchors as *descriptions of the phenomenon* rather than instances,
and give the schema enough coverage (roughly 6–9 per class for a
production schema). The benchmark's low anchor count is a property of the
benchmark, not a recommendation.

The 9-anchor rows come from a self-measuring reference application
(6 classes, 29 labelled messages disjoint from the anchors, `klix_demo.py`
in the sibling `klix-demo` project). It re-computes its own accuracy on
every run, so the figures are reproducible rather than transcribed —
the pattern worth copying when you build your own schema.

**Statistical honesty:** with n=70, differences of 1–2 points between
embedding-based rows are within the 95 % CI (roughly ±9 pts at n=70); the
`klix linear` lead is the only row pair that separates clearly. Treat the
table as directional, not as a ranking with that precision.

**Version note (0.8.1):** the numbers above were re-measured 2026-09-24
with the current FastEmbed release (0.8.x), which computes mean-pooled
MiniLM embeddings (previously CLS pooling). On the fixed 60-case 5-domain
set the klix accuracy is unchanged (nearest 41/60 = 68 %, linear 51/60 =
85 %, verified via `evals/linear_sweep.py`); the `benchmark.py` row for
`klix nearest` moved 76 % → 72 % and `klix linear` 86 % → 84 %, because
that harness includes the GUARD set. Latency was re-measured as well:
the ~10 ms claims of earlier README versions predate the current FastEmbed
release; the forward pass now measures ~50–90 ms on the dev workstation —
still CPU-only and offline, but expect tens of milliseconds per query on
similar hardware.

### vs. SetFit (few-shot training, same examples)

`evals/setfit_baseline.py` — SetFit trains a contrastive few-shot classifier
on the same anchor texts (identical example budget, `num_epochs=1`, same
MiniLM backbone, CPU):

| Dataset | klix (nearest) | klix (linear) | SetFit |
|---|---|---|---|
| HR | 8/12 | 9/12 | 9/12 |
| FIN | 10/12 | 12/12 | 11/12 |
| IMAGE | 9/12 | 11/12 | 11/12 |
| TASK | 7/12 | 11/12 | 10/12 |
| SHOP | 7/12 | 8/12 | 10/12 |
| **total (n=60)** | 41/60 = 68 % | 51/60 = 85 % | **51/60 = 85 %** |

**The honest verdict:** trained few-shot classification (SetFit) *matches*
the klix linear probe (85 %) but buys nothing beyond it — on this benchmark,
training reaches exactly what klix already achieves without any training
step. The klix linear probe and bm25+topk3+coverage configs reach the same
accuracy in sub-100 ms at compile time, with no model artifact, no extra
dependency stack, and full keyword explainability via the sparse channel.
Klix's pitch is therefore NOT "as accurate as training at zero cost" — it is:
equivalent accuracy to few-shot training on these sets, but instant schema
updates (no training step), no saved model per schema, and hard-negative
mining as the training-free accuracy lever (+7 pts measured). SetFit is the
right choice when anchor sets are tiny per class (its contrastive pairing
extracts more from 3-4 anchors); klix is the right choice when schemas
change with the business, the process is iterative, or the deployment must
stay tiny and offline.

### vs. Laya (`convaiinnovations/laya`)

Klix's original inspiration is the trained zero-shot engine
[`laya`](https://pypi.org/project/laya/) (Torch/ModernBERT-large, GPU-oriented).
Measured on the same 70 cases on **CPU**:

| | klix-linear | Laya (zero-shot) |
|---|---|---|
| accuracy | **86 %** | 67 % |
| latency (CPU) | **~13 ms** | ~1.3–1.5 s |
| model size | ~120 MB | ~2 GB |
| setup | anchors (few-shot) | instructions + criteria (zero-shot) |

**The honest trade-off:** Laya needs *no* examples and natively routes 100+
languages with automatic script detection — a real advantage for low-resource
scripts on GPU. Klix is the opposite design point: a tiny model, few-shot anchors
you fully control, and 100× lower CPU latency. They are complements, not
substitutes.

### Choosing a variant

- **`classifier="nearest"` (default)** — most robust with few or mixed-language
  anchors; always a safe baseline.
- **`classifier="linear"`** — highest accuracy on single-language schemas with
  3+ anchors per class; watch for overfitting with mixed-language few-shot data.
- **`classifier="hybrid"`** — learned dense+sparse fusion; wins on keyword-rich
  schemas (asset IDs, SKU codes, error codes). Trails `linear` on plain
  natural-language sets (81.7 % vs 85.0 % at n=60) — opt-in, not a default.
- **`sparse_metric="bm25"`** — BM25 instead of TF-IDF cosine; better when
  anchor lengths vary a lot (+3 pts at n=60 on the nearest path). Combined
  with `topk3 + coverage` it matches `linear` accuracy *without any probe
  training* (85 % at n=60).
- **`label_aggregation="topk"`** — damps single-anchor noise when you have 4+
  anchors per label.
- **Hard negatives** — for production schemas, collect low-confidence live
  decisions (`HardNegativeStore`), review them by hand, attach them as
  counterexamples and recompile. **Honest measured effect (holdout eval,
  `evals/hard_negative_e2e.py`):** on cases the mining step never saw, the
  gain is ≈0 (11/20 → 10/20 on n=20 holdout); an earlier reported +7 pts
  was dominated by memorization of the mining set itself (corrected in
  0.8.1). The workflow is still valuable as a *diagnosis* loop (it surfaces
  which label pairs the schema confuses — fix those by adding/sharpening
  anchors), not as an automatic accuracy lever.

## Evaluation scripts (`evals/`)

`evals/` is a **catalog**, not a test suite (functional tests live in
`tests/`). Every script is runnable from the repo root via
`uv run python -m evals.<name>` (module form, so package imports
`from evals.X import ...` work).
The table distinguishes *live results* (maintained, referenced from this
README/CHANGELOG) from *historical* experiments (kept for reproducibility,
results are snapshots in time — re-running may show different numbers).

| Script | Purpose | Status |
|---|---|---|
| `benchmark.py` | Cross-domain routing (6 domains, 70 cases) — the README table | live |
| `benchmark_bilingual.py` | Bilingual routing (EN/DE, 20 cases) — the README table | live |
| `setfit_baseline.py` | SetFit few-shot comparison (n=60, leakage-verified) | live |
| `hard_negative_e2e.py` | Holdout-verified hard-negative mining effect | live |
| `eval_domains.py` | Labeled 5-domain corpus (60 cases) shared by several evals | live (data source) |
| `variant_sweep.py` | Nearest/topk/coverage sweeps (pre-0.2 history) | historical |
| `corpus_expansion.py` | 273-case paraphrase corpus + bootstrap CIs | historical |
| `expanded_benchmark.py` | The expanded corpus benchmark | historical |
| `linear_sweep.py`, `knob_sweep.py`, `bm25_sweep.py`, `hybrid_sweep.py`, `hybrid_bm25_sweep.py` | Probe / sparse-channel sweeps behind 0.2.x–0.8.0 | historical |
| `charngram_experiment.py` | Did char-n-grams help the probe? (measured: no) | historical |
| `verify_crosslingual.py`, `whatif.py`, `whatif_anchors.py`, `instrument.py` | Feature verification / what-if simulations | historical |
| `eval_baseline.py`, `eval_after.py` | Pre/post head-feature comparisons | historical |
| `bench_batch.py` | Batch-vs-serial latency measurement | historical |
| `bug_hunt.py` | Edge-case hunting script (not a test file) | historical |
| `export_datasets.py` | Export the labeled datasets to JSONL | historical |

**Conventions:** scripts share the labeled datasets via package imports
(`from evals.linear_sweep import ...`, `from evals.eval_domains import
...`). Run them as modules from the repo root (`uv run python -m
evals.<name>`). If you change a shared dataset, re-run the evals that
depend on it.

## Development

```bash
uv sync          # install dependencies
uv run pytest -q # functional test suite (fully offline, no timing gates)
uv run pytest -q -m benchmark  # optional: latency/microbenchmark gates
uv run python examples/demo.py
```

**Test-suite structure:** functional tests (`tests/`) are the CI gate and
never assert wall-clock numbers. Timing-sensitive cases — head evaluation
latency and batch throughput — are marked `@pytest.mark.benchmark` and
excluded from the CI gate, because timing assertions under shared CPU load
are inherently flaky and would mask real failures. Run them explicitly
when you care about latency regressions locally.

**Release chain (automated, no token):** bump the version in `pyproject.toml`
and `__init__.py`, update `CHANGELOG.md`, commit, tag, push — GitHub Actions
builds and publishes to PyPI automatically via **Trusted Publishing (OIDC)**
(workflow `publish.yml`, environment `pypi`; configured once in the PyPI
project settings — no API token is stored in the repo):

```bash
uv build && uv run --with twine python -m twine check dist/*   # pre-tag check
git tag vX.Y.Z
git push origin main vX.Y.Z
```

## Offline / air-gapped deployment

Klix needs no API keys, but the first run downloads the ONNX embedding model
(~120 MB) into the fastembed cache. To prepare an air-gapped machine, cache
the model on a connected machine and transfer it:

```bash
# 1. On a machine with internet: warm the cache once.
uv run python -c "from klix import DecisionEngine; e=DecisionEngine(); e.add_head(__import__('klix').Choice(name='x', options={'a':['alpha']})); e.compile()"

# 2. Find the cache dir (fastembed uses the HF-style local cache):
uv run python -c "from fastembed import TextEmbedding; print(TextEmbedding.list_supported_models()[0]['sources'])"  # model id reference
# cache location (default): ~/.cache/fastembed  (respects XDG_CACHE_HOME / LOCALAPPDATA)

# 3. Copy the cache directory to the air-gapped machine, same path, then
#    set the cache env var if the path differs:
#    XDG_CACHE_HOME=/data/cache   (Linux)
#    LOCALAPPDATA=%CUSTOM_PATH%   (Windows)
```

After that, klix runs fully offline — no network access is ever attempted
at inference time.

## License

MIT
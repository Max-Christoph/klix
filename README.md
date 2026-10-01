# Klix

[![PyPI version](https://img.shields.io/pypi/v/klix-engine.svg)](https://pypi.org/project/klix-engine/)
[![Python](https://img.shields.io/pypi/pyversions/klix-engine.svg)](https://pypi.org/project/klix-engine/)
[![CI](https://github.com/Max-Christoph/klix/actions/workflows/ci.yml/badge.svg)](https://github.com/Max-Christoph/klix/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Max-Christoph/klix/blob/main/LICENSE)

**Lightweight multi-task decision runtime over frozen sentence embeddings for CPU.**  
Evaluates categorical routing, calibrated multi-label scoring, bounded scalar metric projection, and ternary polarity gates over a single shared vector forward pass.

---

## Technical Overview

Klix is a deterministic inference runtime designed for low-latency decision boundaries in automated pipelines and agent architectures. Rather than executing autoregressive language models or orchestrating multiple disjoint estimators, Klix projects input text into a dense embedding space once and evaluates heterogeneous linear decision heads concurrently via vectorized BLAS operations.

* **Inference Complexity:** Embedding cost is O(L · D) via an ONNX-runtime encoder (`paraphrase-multilingual-MiniLM-L12-v2`, 240 MB footprint, D=384). Multi-head evaluation is O(K · D) linear algebra, executing in < 100 µs post-embedding.
* **Latency & Throughput:**
  - *Interactive Single-Item Execution (`decide`):* ~26–30 ms (p50 on a single commodity x86 CPU core).
  - *Vectorized Batch Throughput (`decide_batch`):* ~13.9 ms per document (sustained 72 docs/s at batch sizes >= 64).
* **Out-of-Domain Rejection:** Reject anchor sets and neutral simplex poles allow decision heads to decline classification (`value=None`) when input density falls outside defined class distributions.
* **Symbolic Constraints:** Deterministic regular expression and exact token rules (`Rule`) evaluate as hard boolean constraints alongside vector similarity.
* **Zero PyTorch Runtime Dependency:** Pure CPU execution backed by ONNX Runtime, FastEmbed, NumPy, and scikit-learn.

---

## Scope, Limitations & Non-Goals

To maintain technical clarity and prevent architectural mismatch, Klix explicitly defines its operational boundaries:

1. **Not a Generative Language Model:** Klix does not generate text, extract open entities, or conduct multi-step conversational reasoning. It is strictly an anchor-based decision boundary runtime.
2. **Contrastive Hard Negatives vs. Open-World OOD:** Supplying `reject_anchors` does not solve the unbounded open-set problem (the entire complement embedding space). A reject anchor carves out a localized Voronoi suppression cell against known false positives. Universal open-world Out-of-Domain (OOD) rejection relies on similarity thresholds (e.g. `max_sim < threshold`) or margin gating between leading classes.
3. **Algorithmic Transparency on Linear Probes:** The `linear` classifier mode compiles an L2-regularized multinomial logistic regression probe over frozen representations (essentially FastEmbed + scikit-learn). Klix does not claim a novel optimization solver; its engineering contribution is unified multi-head compilation, single-pass vector reuse, symbolic rule overrides, and low-latency agent graph integration (~26–30 ms interactive, ~13.9 ms batched).
4. **No Backbone Fine-Tuning:** Transformer backbone weights remain strictly frozen. Tasks requiring domain-adapted metric representations should use contrastive fine-tuning frameworks (e.g. SetFit) before exporting to an ONNX runtime.
5. **Matryoshka Truncation Requires MRL Backbones:** Slicing dimensions via `truncate_dim` preserves semantic fidelity only when using models trained with Matryoshka Representation Learning (MRL, e.g. `nomic-embed-text` or `bge-m3`). The default MiniLM model was not trained with MRL; truncating it induces arbitrary geometric distortion.
6. **Calibration Scope for Score and Flag:** While `Choice` and `MultiLabel` are benchmarked on standard public corpora (CLINC150, BANKing77, MASSIVE, GoEmotions), `Score` (polar scalar projection) and `Flag` (3-simplex softmax polarity gate) are parameterized geometric heuristics intended for low-latency state enrichment in agent graphs. Their default calibration settings (`sharpness=12.0`, `center=0.40`, `temperature=1.0`) are starting points that require task-specific calibration.

---

## Architectural Comparison

| Dimension | Custom `scikit-learn` + `sentence-transformers` | `semantic-router` | `SetFit` | `Klix` |
|---|---|---|---|---|
| **Runtime Dependencies** | Heavy (PyTorch, transformers, CUDA/C++) | Lightweight (NumPy, FastEmbed) | Heavy (PyTorch, transformers) | **Lightweight** (ONNX Runtime, NumPy, scikit-learn) |
| **Model Footprint** | ~800 MB (PyTorch CPU runtime) | ~240 MB on-disk ONNX | ~800 MB (PyTorch CPU runtime) | **~240 MB on-disk ONNX** |
| **Evaluation Scope** | Typically single-task pipelines | Single-label route thresholding | Fine-tuned single-task classifier | **Multi-task:** Single-label, Multi-label, 1D Scalar, Ternary Flag |
| **Pass Architecture** | Manual vector passing across separate estimators | Single vector pass for routing | Single vector pass per model | **Single forward pass shared across all N heads** |
| **Training Paradigm** | Manual fit loop per task | Parameter-free anchor matching | Contrastive fine-tuning (backpropagation) | **Frozen backbone; parameter-free centroids or convex L2 probe** |
| **Rejection Mechanism** | Manual probability thresholding | Distance thresholding | Manual probability thresholding | **Dual rejection:** Explicit negative poles + cosine thresholds |
| **Symbolic Overrides** | External wrapper required | External wrapper required | External wrapper required | **Native compiled hard/boost rules (`Rule`)** |

---

## Decision Head Architecture

Each query is vectorized once into a 384-dimensional normalized vector `q` (`||q|| = 1`). All attached heads evaluate concurrently via linear algebra. For complete mathematical formulations and LaTeX proofs, see [docs/MATHEMATICS.md](docs/MATHEMATICS.md).

### 1. Categorical Choice (`Choice`)
* **Centroid (`classifier="centroid"`, Default):** Computes class centroids `c_k = mean(anchors_k) / ||mean(anchors_k)||`. Inference evaluates `argmax(q · c_k)` in O(K · D) floating-point operations.
* **Out-of-Domain Rejection:** Evaluates queries against contrastive `reject_anchors` (rejects when `reject_score > best_score`) and minimum cosine thresholds (`min_confidence`).
* **Linear Probe (`classifier="linear"`):** Compiles an L2-regularized logistic regression boundary over anchor embeddings at compile time.
* **Hybrid (`classifier="hybrid"`):** Fuses dense cosine similarity with sparse BM25 token matching for exact identifier support.

### 2. Calibrated Multi-Label (`MultiLabel`)
Evaluates each category as an independent decision boundary. Cosine similarity `a_k = q · c_k` is calibrated onto [0, 1] via a sigmoid transformation `s_k = 1 / (1 + exp(-sharpness * (a_k - center)))`. Active labels are selected via threshold truncation (`s_k >= threshold`).

### 3. Continuous Metric Projection (`Score`)
Projects a query onto a bounded continuous scale `[min_val, max_val]` based on its relative proximity between low and high anchor distributions.

### 4. Ternary Polarity Flag (`Flag`)
Evaluates binary classification against an explicit neutral ambiguity pole over a 3-component probability simplex via temperature-scaled softmax. Returns `True`, `False`, or `None` (when neutral dominates or confidence is below threshold).

---

## Installation

```bash
pip install klix-engine
# Optional LangChain/LangGraph integration:
pip install "klix-engine[langchain]"
```

*Note: On initial invocation, FastEmbed downloads and caches the multilingual ONNX model (`paraphrase-multilingual-MiniLM-L12-v2`, ~240 MB) locally. Subsequent runs execute entirely offline.*

---

## Minimal Example

```python
from klix import DecisionEngine, Choice, MultiLabel, Score, Flag

engine = DecisionEngine()

# 1. Categorical routing with out-of-domain rejection pole
engine.add_head(
    Choice(
        name="queue",
        options={
            "it_ops":   ["VPN down", "server unreachable", "laptop won't boot"],
            "ot_plant": ["robot cell stopped", "PLC fault", "conveyor belt stalled"],
            "finance":  ["cost center over budget", "invoice approval needed"],
        },
        reject_anchors=["casual coffee chat", "lunch plans"],  # out-of-domain -> value=None
    )
)

# 2. Independent calibrated multi-label scoring
engine.add_head(
    MultiLabel(
        name="tags",
        options={
            "hardware": ["laptop broken", "PLC hardware failure", "cable snapped"],
            "critical": ["production halted", "acute danger", "emergency stop"],
            "network":  ["DNS error", "wifi disconnected", "packet loss"],
        },
        threshold=0.5,
    )
)

# 3. Continuous scalar metric axis
engine.add_head(
    Score(
        name="urgency",
        low_anchors=["routine maintenance", "whenever you have time"],
        high_anchors=["immediate production stop", "acute critical hazard"],
        min_val=0.0,
        max_val=5.0,
    )
)

# 4. Binary flag with neutral rejection pole
engine.add_head(
    Flag(
        name="is_security",
        true_anchors=["ransomware detected", "credential leak", "hacker attack"],
        false_anchors=["ordinary hardware defect", "routine update"],
        neutral_anchors=["general question", "support inquiry"],
    )
)

engine.compile()

# Evaluate single query
res = engine.decide("PLC-34 reports critical hardware error, production halted!")

print(res.queue)                       # 'ot_plant'
print(res.tags)                        # ['critical', 'hardware'] (scores >= threshold)
print(res.details("tags")["scores"])   # {'critical': 0.94, 'hardware': 0.88, 'network': 0.12}
print(res.urgency)                     # 4.82
print(res.is_security)                 # False (probability=0.04)
```

---

## Empirical Benchmarks & Methodological Analysis

All evaluations were executed on commodity x86 CPU hardware without GPU acceleration. Evaluation harnesses, test split integrity assertions, and JSON artifacts are archived in [`docs/BENCHMARKS.md`](https://github.com/Max-Christoph/klix/blob/main/docs/BENCHMARKS.md).

### Summary of Empirical Results

| Benchmark / Corpus | Task Type | Classes (K) | Prior (1/K) | Test Split (N) | Klix Centroid | External Baseline / Reference | Sustained Throughput |
|---|---|:---:|:---:|:---:|---|---|:---:|
| **CLINC150** | Intent + OOD Rejection | 150 | 0.7 % | 5,500 *(full)* | **86.9 % Acc / 94.8 % AUROC** *(k=10)*<br>78.8 % Acc / 92.9 % AUROC *(k=3)* | 86.2 % Acc / 92.3 % AUROC *(sklearn LogReg, k=10)*<br>76.6 % Acc / 89.1 % AUROC *(sklearn LogReg, k=3)*<br>73.0 % Acc / 91.4 % AUROC *(semantic-router, k=3)* | **99 docs/s** |
| **BANKing77** | Intent Routing | 77 | 1.3 % | 3,080 *(full)* | **61.4 % Acc** *(k=3 anchors)* | 53.3 % *(canonical labels)* | **72 docs/s** |
| **MASSIVE** (English) | Intent Routing | 60 | 1.7 % | 2,974 *(full)* | **57.3 % Acc** *(k=3 + label centroid)* | 47.9 % *(canonical labels)* | **72 docs/s** |
| **MASSIVE** (German) | Intent Routing | 60 | 1.7 % | 2,974 *(full)* | **44.4 % Acc** *(k=3 + label centroid)* | 40.5 % *(Qwen-2B LLM)* | **72 docs/s** |
| **GoEmotions** | Multi-Label (28 Affects) | 28 | 3.6 % | 500 *(subsample)* | **19.5 % Micro-F1** *(k=10 centroid)* | 9.8 % F1 *(k=3 anchors)* | **55 docs/s** |
| **Cross-Domain Routing** | Intent Routing | 6 | 16.7 % | 70 *(sanity check)* | **84.3 % Acc** *(centroid / linear)* | 78.0 % *(Embed-KNN)* | **72 docs/s** |
| **Few-Shot vs. Fine-Tuning** | Classification Probe | 5 | 20.0 % | 60 *(sanity check)* | **85.0 % Acc** *(linear probe)* | 85.0 % *(SetFit trained)* | **72 docs/s** |

---

### Methodological Observations & Ablations

1. **Out-of-Scope (OOD) Gating & External Baselines on CLINC150 (K=150 Intents + 1,000 OOS Queries):**  
   Evaluated on the full 5,500-sample benchmark (Larson et al., EMNLP 2019). Parameter-free centroid projection reaches **86.93% in-scope accuracy** and **94.79% AUROC** (k=10), performing competitively with an L2-regularized multinomial `LogisticRegression` baseline on identical embeddings (86.22% Acc, 92.28% AUROC) while compiling in < 0.1 s vs. 120 s. Compared to 1-nearest-neighbor anchor matching (`semantic-router` equivalent: 73.02% Acc, 91.44% AUROC at k=3), centroid aggregation improves accuracy by +5.76 percentage points while evaluating in O(K · D) flops. At matched operating points, supplying 100 training OOS examples as reject anchors blocks 53.8% of unseen OOS test queries with only 2.24% in-scope false rejection (101/4,500), lifting combined OOD rejection to 79.1% at the standard 95% in-scope retention threshold (+2.1 pt over pure thresholding).

2. **Anchor-Label Hybridization on MASSIVE (K=60):**  
   While raw arbitrary few-shot sentence sampling alone (k=3, 43.8% on English) scores lower than bare curated label names (47.9%), combining the canonical label name into the anchor pool under centroid aggregation anchors the semantic subspace: accuracy jumps to **57.3%** on English (+13.5 pt) and **44.4%** on German (+8.2 pt), outperforming the local 2B parameter autoregressive LLM baseline (40.5%).

3. **Fine-Grained Multi-Label Classification with Frozen Backbones (GoEmotions Corpus):**  
   GoEmotions evaluates 28 fine-grained affective categories with substantial label co-occurrence and semantic overlap (e.g., *admiration* vs. *approval* vs. *pride*). Without domain-specific metric fine-tuning, a frozen 384-dimensional representation attains a Micro-F1 of **19.5%** (k=10 centroid, Macro-F1 18.1%) against a random prior of 3.6%. Centroid aggregation outperforms 1-nearest-neighbor matching (14.1% Micro-F1) by +5.4 percentage points while reducing inference to a single BLAS matrix-vector product.

4. **Sample Size Scope:**  
   The cross-domain (N=70) and SetFit ablation (N=60) test sets represent small qualitative smoke tests for domain-specific schemas. For high-cardinality statistical characterization, primary reference should be made to the full public test splits of CLINC150 (N=5,500, K=150), BANKing77 (N=3,080, K=77), and MASSIVE (N=2,974, K=60).

5. **Selective Classification (Risk-Coverage Trade-Off in Agent Routing):**  
   In agent dispatch architectures, low-confidence or out-of-domain queries return `None` to escalate to an LLM fallback. On CLINC150 (k=10, N=5,500), sweeping the cosine threshold demonstrates this empirical risk-coverage trade-off: at threshold tau = 0.55, coverage is 81.3% with 88.9% in-scope accuracy and 77.9% of OOS queries blocked; at tau = 0.65, coverage is 71.7% with 90.5% in-scope accuracy and 89.4% of OOS queries blocked. High-confidence routing resolves in ~26–30 ms per query.

---

### Anchor Density Scaling (BANKing77, 77 Classes)

Evaluated on a deterministic 500-sample stride subsample of BANKing77 (k=3 nearest scores 60.2% on this subsample vs. 61.4% on the full 3,080 test split):

| Classifier Mode | Anchors per Class (k) | Compilation Time | Test Accuracy | Per-Item Latency |
|---|:---:|:---:|:---:|:---:|
| `nearest` | k=3 | 4.7 s | 60.2 % | 29.7 ms/doc |
| `centroid` | k=3 | 5.3 s | **64.8 %** (+4.6 pt) | 26.6 ms/doc |
| `linear` | k=3 | 10.5 s | **67.8 %** (+7.6 pt) | 26.1 ms/doc |
| `centroid` | k=10 | 22.4 s | **76.6 %** (+16.4 pt) | 31.3 ms/doc |
| `linear` | k=10 | 85.0 s | **79.8 %** (+19.6 pt) | 28.1 ms/doc |
| `centroid` | k=20 | 39.6 s | **79.6 %** (+19.4 pt) | 26.7 ms/doc |
| `linear` | k=20 | 132 s | **85.2 %** (+25.0 pt) | 41.9 ms/doc |

*Observation:* Centroid projection consistently improves accuracy over nearest-neighbor matching by +4.6 to +19.4 percentage points while maintaining O(K · D) computational complexity. Regularized linear probes reach 85.2% accuracy when anchor density satisfies k >= 10 examples per class.

---

## Configuration Reference

| Parameter | Scope | Type | Description |
|---|---|---|---|
| `options` | `Choice`, `MultiLabel` | `dict[str, list[str]]` | Mapping of class identifier to anchor sentence corpus. |
| `classifier` | `Choice`, `MultiLabel` | `str` | Inference mode: `"centroid"` (default), `"linear"`, `"nearest"`, `"hybrid"`. |
| `threshold` | `MultiLabel`, `Flag` | `float` | Minimum decision threshold for active category membership or boolean assertion. |
| `sharpness` | `MultiLabel` | `float` | Logistic steepness parameter `sharpness` (gamma) for calibrated sigmoid scoring (default: `12.0`). |
| `center` | `MultiLabel` | `float` | Cosine similarity inflection center `center` (c_0) for sigmoid calibration (default: `0.40`). |
| `calibration` | `MultiLabel` | `str` | Score mapping: `"sigmoid"` (default), `"linear"`, `"cosine"`. |
| `reject_anchors` | `Choice` | `list[str]` | Targeted contrastive hard negatives; dominant similarity resolves to `value=None`. |
| `neutral_anchors` | `Flag` | `list[str]` | Ambiguity pole on the 3-simplex; dominant similarity resolves to `value=None`. |
| `rules` | `Choice` | `list[Rule]` | Deterministic symbolic constraints: `Rule(label=..., any_of=[...], mode="force"|"boost")`. |
| `sparse_fastpath` | `DecisionEngine` | `bool` | Deterministic exact keyword index bypassing embedding inference when unambiguous (< 1 ms). |
| `glossary` | `Choice`, `DecisionEngine` | `dict` | Pre-compiled cross-lingual lexical synonym map. |
| `truncate_dim` | `DecisionEngine` | `int` | Matryoshka prefix dimension truncation. Note: requires MRL-trained backbone (e.g. `nomic-embed-text`). |

---

## Vectorized Batch Inference

To evaluate document corpora without per-item framework overhead:

```python
texts = ["First query document...", "Second query document...", ...]
results = engine.decide_batch(texts)  # Vectorized ONNX forward pass; sustained 72 docs/s
```

---

## LangChain & LangGraph Integration

Klix provides a native low-latency routing and state-enrichment layer for agent execution graphs:

```python
from klix.integrations.langchain import KlixRouterRunnable, create_klix_router

# 1. LangChain Runnable: returns route key or enriches state dictionary
router = KlixRouterRunnable(engine=engine, route_head="queue", enrich_state=True)
chain = router | RunnableBranch(...)

# 2. LangGraph Conditional Edge: low-latency (< 30 ms) graph routing
workflow.add_conditional_edges(
    "supervisor",
    create_klix_router(engine, head_name="queue"),
    {
        "it_ops": "it_agent",
        "finance": "finance_agent",
        None: "fallback_llm",
    },
)
```

See [`examples/langchain_agent_router.py`](https://github.com/Max-Christoph/klix/blob/main/examples/langchain_agent_router.py) for the executable reference pipeline.

---

## License

This project is licensed under the MIT License. See [LICENSE](https://github.com/Max-Christoph/klix/blob/main/LICENSE) for details.
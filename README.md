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

* **Inference Complexity:** Embedding cost is $O(L \cdot D)$ via an ONNX-runtime encoder (`paraphrase-multilingual-MiniLM-L12-v2`, 240 MB footprint, $D=384$). Multi-head evaluation is $O(K \cdot D)$ linear algebra, executing in $< 100\,\mu\text{s}$ post-embedding.
* **Latency & Throughput:**
  - *Interactive Single-Item Execution (`decide`):* ~26–30 ms (p50 on a single commodity x86 CPU core).
  - *Vectorized Batch Throughput (`decide_batch`):* ~13.9 ms per document (sustained 72 docs/s at batch sizes $\ge 64$).
* **Out-of-Domain Rejection:** Reject anchor sets and neutral simplex poles allow decision heads to decline classification (`value=None`) when input density falls outside defined class distributions.
* **Symbolic Constraints:** Deterministic regular expression and exact token rules (`Rule`) evaluate as hard boolean constraints alongside vector similarity.
* **Zero PyTorch Runtime Dependency:** Pure CPU execution backed by ONNX Runtime, FastEmbed, NumPy, and scikit-learn.

---

## Scope, Limitations & Non-Goals

To maintain technical clarity and prevent architectural mismatch, Klix explicitly defines its operational boundaries:

1. **Not a Generative Language Model:** Klix does not generate text, extract open entities, or conduct multi-step conversational reasoning. It is strictly an anchor-based decision boundary runtime.
2. **Contrastive Hard Negatives vs. Open-World OOD:** Supplying `reject_anchors` does not solve the unbounded open-set problem ($\mathbb{R}^D \setminus \bigcup \mathcal{C}_k$). A reject anchor carves out a localized Voronoi suppression cell against known false positives. Universal open-world Out-of-Domain (OOD) rejection relies on similarity thresholds ($\max_k s_k < \tau_{\min}$) or margin gating between leading classes.
3. **Algorithmic Transparency on Linear Probes:** The `linear` classifier mode compiles an $L_2$-regularized multinomial logistic regression probe over frozen representations (essentially FastEmbed + scikit-learn). Klix does not claim a novel optimization solver; its engineering contribution is unified multi-head compilation, single-pass vector reuse, symbolic rule overrides, and sub-15ms agent graph integration.
4. **No Backbone Fine-Tuning:** Transformer backbone weights remain strictly frozen. Tasks requiring domain-adapted metric representations should use contrastive fine-tuning frameworks (e.g. SetFit) before exporting to an ONNX runtime.
5. **Matryoshka Truncation Requires MRL Backbones:** Slicing dimensions via `truncate_dim` preserves semantic fidelity only when using models trained with Matryoshka Representation Learning (MRL, e.g. `nomic-embed-text` or `bge-m3`). The default MiniLM model was not trained with MRL; truncating it induces arbitrary geometric distortion.

---

## Architectural Comparison

| Dimension | Custom `scikit-learn` + `sentence-transformers` | `semantic-router` | `SetFit` | `Klix` |
|---|---|---|---|---|
| **Runtime Dependencies** | Heavy (PyTorch, transformers, CUDA/C++) | Lightweight (NumPy, FastEmbed) | Heavy (PyTorch, transformers) | **Lightweight** (ONNX Runtime, NumPy, scikit-learn) |
| **Model Footprint** | ~2–4 GB memory allocation | ~240 MB on-disk ONNX | ~2–4 GB memory allocation | **~240 MB on-disk ONNX** |
| **Evaluation Scope** | Typically single-task pipelines | Single-label route thresholding | Fine-tuned single-task classifier | **Multi-task:** Single-label, Multi-label, 1D Scalar, Ternary Flag |
| **Pass Architecture** | Repeated forward passes for multiple heads | Single vector pass for routing | Single vector pass per model | **Single forward pass shared across all $N$ heads** |
| **Training Paradigm** | Manual fit loop per task | Parameter-free anchor matching | Contrastive fine-tuning (backpropagation) | **Frozen backbone; parameter-free centroids or convex $L_2$ probe** |
| **Rejection Mechanism** | Manual probability thresholding | Distance thresholding | Manual probability thresholding | **Dual rejection:** Explicit negative poles + cosine thresholds |
| **Symbolic Overrides** | External wrapper required | External wrapper required | External wrapper required | **Native compiled hard/boost rules (`Rule`)** |

---

## Mathematical Formulation

Let $\mathbf{q} \in \mathbb{R}^D$ denote the $L_2$-normalized dense query representation ($\|\mathbf{q}\|_2 = 1$) produced by the shared embedding backbone. Let $\{\mathbf{x}_{k,1}, \dots, \mathbf{x}_{k,n_k}\}$ denote the normalized anchor embeddings provided for class $k \in \{1, \dots, K\}$.

### 1. Categorical Choice (`Choice`)

#### Mode: Centroid (`classifier="centroid"`, Default)
The class representative $\mathbf{c}_k$ is the normalized geometric mean of its constituent anchor vectors:
$$\mathbf{c}_k = \frac{\sum_{i=1}^{n_k} \mathbf{x}_{k,i}}{\|\sum_{i=1}^{n_k} \mathbf{x}_{k,i}\|_2}$$
Cosine similarities across all $K$ classes are computed via a single matrix-vector product $\mathbf{s} = \mathbf{C}\mathbf{q}$, where $\mathbf{C} \in \mathbb{R}^{K \times D}$. The optimal label is selected via:
$$k^* = \arg\max_{k \in \{1,\dots,K\}} (\mathbf{q} \cdot \mathbf{c}_k)$$

#### Out-of-Domain Rejection
Given an optional set of reject anchors $\mathbf{R} = \{\mathbf{r}_1, \dots, \mathbf{r}_M\}$, the decision is nullified (`value=None`) if the maximum similarity to any rejection anchor exceeds the target class similarity, or if similarity fails a minimum threshold $\tau_{\min}$:
$$\text{output} = \begin{cases} \text{None}, & \text{if } \max_{m} (\mathbf{q} \cdot \mathbf{r}_m) > \mathbf{q} \cdot \mathbf{c}_{k^*} \;\lor\; (\mathbf{q} \cdot \mathbf{c}_{k^*}) < \tau_{\min} \\ k^*, & \text{otherwise} \end{cases}$$

#### Mode: Linear Probe (`classifier="linear"`)
Fits an $L_2$-regularized multinomial logistic regression boundary directly over anchor representations at compile time:
$$k^* = \arg\max_{k \in \{1,\dots,K\}} (\mathbf{w}_k^T \mathbf{q} + b_k), \quad \text{subject to } \min_{\mathbf{W}, \mathbf{b}} \mathcal{L}_{\text{CE}}(\mathbf{W}, \mathbf{b}) + \frac{\lambda}{2} \|\mathbf{W}\|_F^2$$

#### Mode: Hybrid Sparse-Dense (`classifier="hybrid"`)
Blends dense cosine similarity with sparse BM25 scores over anchor lexical tokens:
$$S(q, k) = \alpha (\mathbf{q} \cdot \mathbf{c}_k) + (1 - \alpha) S_{\text{BM25}}(q, k)$$

---

### 2. Calibrated Multi-Label Scoring (`MultiLabel`)

Unlike softmax-based categorical routing, `MultiLabel` treats each category as an independent decision boundary. The raw cosine alignment $a_k = \mathbf{q} \cdot \mathbf{c}_k$ is mapped onto a calibrated probability $s_k \in [0, 1]$ via a parameterized sigmoid transformation:
$$s_k(\mathbf{q}) = \sigma\left(\gamma \cdot (a_k - c_0)\right) = \frac{1}{1 + \exp\left(-\gamma \cdot (a_k - c_0)\right)}$$
where $\gamma$ denotes the calibration sharpness (`sharpness=12.0`) and $c_0$ denotes the cosine inflection center (`center=0.40`). The set of active labels is defined by threshold truncation:
$$\mathcal{Y}(\mathbf{q}) = \{k \mid s_k(\mathbf{q}) \ge \tau\}$$

---

### 3. Continuous Metric Projection (`Score`)

Maps a query onto a bounded continuous interval $[V_{\min}, V_{\max}]$ based on its relative proximity to two polar anchor distributions $\mathcal{A}_{\text{low}}$ and $\mathcal{A}_{\text{high}}$:
$$v(\mathbf{q}) = V_{\min} + (V_{\max} - V_{\min}) \cdot \frac{\text{top}_k(\mathbf{q}, \mathcal{A}_{\text{high}}) - \text{top}_k(\mathbf{q}, \mathcal{A}_{\text{low}}) + 1}{2}$$

---

### 4. Ternary Polarity Flag (`Flag`)

Evaluates binary classification against an explicit neutral rejection pole over a 3-component simplex:
$$P(y) = \frac{\exp((\mathbf{q} \cdot \mathbf{c}_y) / T)}{\sum_{j \in \{\text{true}, \text{false}, \text{neutral}\}} \exp((\mathbf{q} \cdot \mathbf{c}_j) / T)}, \quad y \in \{\text{true}, \text{false}, \text{neutral}\}$$
The output resolves to boolean `True` or `False` if $P(y) \ge \tau$. If the neutral pole dominates ($P(\text{neutral}) > \max(P(\text{true}), P(\text{false}))$), the head returns `None`.

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

| Benchmark / Corpus | Task Type | Classes ($K$) | Prior ($1/K$) | Test Split ($N$) | Klix Performance | Baseline / Reference | Sustained Throughput |
|---|---|:---:|:---:|:---:|---|---|:---:|
| **BANKing77** | Intent Routing | 77 | 1.3 % | 3,080 *(full)* | **61.4 % Acc** *(k=3 anchors)* | 53.3 % *(canonical labels)* | **72 docs/s** |
| **MASSIVE** (English) | Intent Routing | 60 | 1.7 % | 2,974 *(full)* | **43.8 % Acc** *(k=3 anchors)* | 47.9 % *(canonical labels)* | **72 docs/s** |
| **MASSIVE** (German) | Intent Routing | 60 | 1.7 % | 2,974 *(full)* | **36.2 % Acc** *(k=3 anchors)* | 40.5 % *(Qwen-2B LLM)* | **72 docs/s** |
| **GoEmotions** | Multi-Label (28 Affects) | 28 | 3.6 % | 500 *(subsample)* | **19.5 % Micro-F1** *(k=10 centroid)* | 9.8 % F1 *(k=3 anchors)* | **55 docs/s** |
| **Cross-Domain Routing** | Intent Routing | 6 | 16.7 % | 70 *(sanity check)* | **84.3 % Acc** *(centroid / linear)* | 78.0 % *(Embed-KNN)* | **72 docs/s** |
| **Few-Shot vs. Fine-Tuning** | Classification Probe | 5 | 20.0 % | 60 *(sanity check)* | **85.0 % Acc** *(linear probe)* | 85.0 % *(SetFit trained)* | **72 docs/s** |

---

### Methodological Observations & Limitations

1. **Anchor Noise vs. Canonical Label Semantics (MASSIVE Corpus):**  
   On the 60-class MASSIVE benchmark, using $k=3$ arbitrary anchor sentences per class yields **43.8%** accuracy, whereas using bare, human-curated label names (e.g., `alarm_set`, `datetime_query`) achieves **47.9%**. This demonstrates that arbitrary few-shot sentence sampling without outlier filtering introduces intra-class variance in a frozen embedding space, whereas concise canonical label descriptors align tightly with pre-trained lexical associations.

2. **Fine-Grained Multi-Label Classification with Frozen Backbones (GoEmotions Corpus):**  
   GoEmotions evaluates 28 fine-grained affective categories with substantial label co-occurrence and semantic overlap (e.g., *admiration* vs. *approval* vs. *pride*). Without domain-specific metric fine-tuning, a frozen 384-dimensional representation attains a Micro-F1 of **19.5%** ($k=10$ centroid, Macro-F1 18.1%) against a random prior of 3.6%. Centroid aggregation outperforms 1-nearest-neighbor matching (14.1% Micro-F1) by +5.4 percentage points while reducing inference to a single BLAS matrix-vector product.

3. **Sample Size Scope:**  
   The cross-domain ($N=70$) and SetFit ablation ($N=60$) test sets represent small qualitative smoke tests for domain-specific schemas. For high-cardinality statistical characterization, primary reference should be made to the full public test splits of BANKing77 ($N=3,080$, $K=77$) and MASSIVE ($N=2,974$, $K=60$).

4. **Selective Classification (The Reject Option in Agent Routing):**  
   In agent dispatch architectures, an unconstrained multi-class error rate over 60–77 classes does not directly map to pipeline failures. By configuring rejection thresholds or explicit `reject_anchors`, the system operates in a selective classification regime: high-confidence queries ($\ge 0.80$, exhibiting 90–98% precision on bounded action domains) are routed immediately in $< 15\,\text{ms}$, while low-confidence or out-of-domain instances return `None` and escalate to an autoregressive LLM fallback.

---

### Anchor Density Scaling (BANKing77, 77 Classes)

Evaluated on a deterministic 500-sample stride subsample of BANKing77 ($k=3$ nearest scores 60.2% on this subsample vs. 61.4% on the full 3,080 test split):

| Classifier Mode | Anchors per Class ($k$) | Compilation Time | Test Accuracy | Per-Item Latency |
|---|:---:|:---:|:---:|:---:|
| `nearest` | $k=3$ | 4.7 s | 60.2 % | 29.7 ms/doc |
| `centroid` | $k=3$ | 5.3 s | **64.8 %** (+4.6 pt) | 26.6 ms/doc |
| `linear` | $k=3$ | 10.5 s | **67.8 %** (+7.6 pt) | 26.1 ms/doc |
| `centroid` | $k=10$ | 22.4 s | **76.6 %** (+16.4 pt) | 31.3 ms/doc |
| `linear` | $k=10$ | 85.0 s | **79.8 %** (+19.6 pt) | 28.1 ms/doc |
| `centroid` | $k=20$ | 39.6 s | **79.6 %** (+19.4 pt) | 26.7 ms/doc |
| `linear` | $k=20$ | 132 s | **85.2 %** (+25.0 pt) | 41.9 ms/doc |

*Observation:* Centroid projection consistently improves accuracy over nearest-neighbor matching by +4.6 to +19.4 percentage points while maintaining $O(K \cdot D)$ computational complexity. Regularized linear probes reach 85.2% accuracy when anchor density satisfies $k \ge 10$ examples per class.

---

## Configuration Reference

| Parameter | Scope | Type | Description |
|---|---|---|---|
| `options` | `Choice`, `MultiLabel` | `dict[str, list[str]]` | Mapping of class identifier to anchor sentence corpus. |
| `classifier` | `Choice`, `MultiLabel` | `str` | Inference mode: `"centroid"` (default), `"linear"`, `"nearest"`, `"hybrid"`. |
| `threshold` | `MultiLabel`, `Flag` | `float` | Minimum decision threshold for active category membership or boolean assertion. |
| `sharpness` | `MultiLabel` | `float` | Logistic steepness parameter $\gamma$ for calibrated sigmoid scoring (default: `12.0`). |
| `center` | `MultiLabel` | `float` | Cosine similarity inflection center $c_0$ for sigmoid calibration (default: `0.40`). |
| `calibration` | `MultiLabel` | `str` | Score mapping: `"sigmoid"` (default), `"linear"`, `"cosine"`. |
| `reject_anchors` | `Choice` | `list[str]` | Targeted contrastive hard negatives; dominant similarity resolves to `value=None`. |
| `neutral_anchors` | `Flag` | `list[str]` | Ambiguity pole on the 3-simplex; dominant similarity resolves to `value=None`. |
| `rules` | `Choice` | `list[Rule]` | Deterministic symbolic constraints: `Rule(label=..., any_of=[...], mode="force"|"boost")`. |
| `sparse_fastpath` | `DecisionEngine` | `bool` | Deterministic exact keyword index bypassing embedding inference when unambiguous ($< 1\,\text{ms}$). |
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

# 2. LangGraph Conditional Edge: low-latency (< 15 ms) graph routing
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
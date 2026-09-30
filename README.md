# Klix

[![PyPI version](https://img.shields.io/pypi/v/klix-engine.svg)](https://pypi.org/project/klix-engine/)
[![Python](https://img.shields.io/pypi/pyversions/klix-engine.svg)](https://pypi.org/project/klix-engine/)
[![CI](https://github.com/Max-Christoph/klix/actions/workflows/ci.yml/badge.svg)](https://github.com/Max-Christoph/klix/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Fast, zero-training semantic decisions on CPU. Route text, assign multi-label scores, extract boolean flags, and score continuous axes — by providing example sentences instead of training models.**

---

- **High Throughput on CPU:** **~13.9 ms/doc** in bulk (72 docs/s sustained → **100,000 documents in ~24 minutes** on a standard laptop CPU).
- **Accurate & Scalable:** **61.4 %** on 77 classes with minimal anchors ($k=3$), scaling up to **85.2 %** with $k=20$ anchors on BANKing77.
- **Four Decoupled Decision Heads:** `Choice` (single-label routing), `MultiLabel` (calibrated continuous scores $[0, 1]$), `Score` (1D axis), and `Flag` (boolean confidence).
- **Self-Contained & Offline:** 240 MB on-disk multilingual ONNX model (`paraphrase-multilingual-MiniLM-L12-v2`). No GPU required, 0 € API costs.
- **Deterministic Guardrails:** Layer exact business rules (`Rule`) and cross-lingual synonym bridges (`Glossary`) over vector geometry without fine-tuning.

---

## Installation

```bash
pip install klix-engine
# or with uv
uv add klix-engine
```

*On first import, FastEmbed caches the multilingual ONNX model (240 MB on disk). Afterwards, everything runs 100 % offline.*

---

## Quickstart

```python
from klix import DecisionEngine, Choice, MultiLabel, Score, Flag

engine = DecisionEngine()

# 1. Single-label routing
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

# 2. Multi-label classification with continuous scores [0.0, 1.0]
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

# 3. Continuous 1D metric axis
engine.add_head(
    Score(
        name="urgency",
        low_anchors=["routine maintenance", "whenever you have time"],
        high_anchors=["immediate production stop", "acute critical hazard"],
        min_val=0.0,
        max_val=5.0,
    )
)

# 4. Boolean flag with neutral rejection pole
engine.add_head(
    Flag(
        name="is_security",
        true_anchors=["ransomware detected", "credential leak", "hacker attack"],
        false_anchors=["ordinary hardware defect", "routine update"],
        neutral_anchors=["general question", "support inquiry"],
    )
)

engine.compile()

# Evaluate text in milliseconds
res = engine.decide("PLC-34 reports critical hardware error, production halted!")

print(res)
# <DecisionResult (46.2ms): queue=ot_plant, tags=['critical', 'hardware'], urgency=4.8, is_security=False>

print(res.queue)                       # 'ot_plant'
print(res.tags)                        # ['critical', 'hardware'] (categories with score >= 0.5)
print(res.details("tags")["scores"])   # {'critical': 0.94, 'hardware': 0.88, 'network': 0.12}
print(res.urgency)                     # 4.82
print(res.is_security)                 # False (probability=0.04)
```

---

## The Four Decision Heads

Each text is embedded **exactly once** by the shared semantic backbone; each head then evaluates in microseconds:

| Head | Purpose | Output (`res.<name>`) | Key Knobs |
|---|---|---|---|
| [`Choice`](#choosing-a-variant--three-rules) | Single-label classification & routing | Winning label string (or `None`) | `classifier="centroid"`, `reject_anchors`, `keyword_boost` |
| **`MultiLabel`** | Multi-label classification with continuous scores | List of active label strings $\ge$ threshold | `threshold=0.5`, `sharpness=12.0`, `center=0.40`, `calibration="sigmoid"` |
| `Score` | Calibrated continuous metric axis (e.g. 0 to 5) | Float in `[min_val, max_val]` | `aggregation="topk"`, `min_coverage` |
| `Flag` | Binary boolean decision with neutral pole | `True`, `False`, or `None` | `threshold=0.5`, `temp=0.12`, `neutral_anchors` |

---

## Performance & Benchmarks

Every metric is measured on a standard Intel CPU without a GPU. Full provenance and test setups are documented in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

### At a Glance

| Benchmark / Task | Classes | Test Cases | Klix Result | Baseline | Bulk Throughput |
|---|:---:|:---:|:---:|:---:|:---:|
| **BANKing77** *(77 banking intents)* | 77 | 3,080 | **61.4 %** *(k=3 anchors)* | 53.3 % *(bare labels)* | **72 docs/s** |
| **MASSIVE** *(Amazon intent, EN)* | 60 | 2,974 | **43.8 %** *(k=3 anchors)* | 47.9 % *(bare labels)* | **72 docs/s** |
| **MASSIVE** *(Amazon intent, DE)* | 60 | 2,974 | **36.2 %** *(k=3 anchors)* | 40.5 % *(Qwen 2B LLM)* | **72 docs/s** |
| **Cross-Domain Routing** *(6 domains)* | 6 | 70 | **84.3 %** *(centroid / linear)* | 78.0 % *(dense Embed-KNN)* | **72 docs/s** |
| **Few-Shot vs. Training** | 5 | 60 | **85.0 %** *(linear probe)* | 85.0 % *(SetFit contrastive)* | **72 docs/s** |

### Accuracy Scales with Anchor Density (BANKing77, 77 classes)

On complex, fine-grained taxonomies (BANKing77 has 77 distinct classes, random chance = 1.3%), accuracy scales directly with anchor quality and classifier choice on the exact same CPU:

| Classifier | Anchors / class ($k$) | Compile Time | Accuracy | Bulk Latency |
|---|:---:|:---:|:---:|:---:|
| `nearest` | $k=3$ | 4.7 s | 60.2 % | 29.7 ms/doc |
| `centroid` | $k=3$ | 5.3 s | **64.8 %** *(+4.6 pt)* | 26.6 ms/doc |
| `linear` | $k=3$ | 10.5 s | **67.8 %** *(+7.6 pt)* | 26.1 ms/doc |
| `centroid` | $k=10$ | 22.4 s | **76.6 %** *(+16.4 pt)* | 31.3 ms/doc |
| `linear` | $k=10$ | 85.0 s | **79.8 %** *(+19.6 pt)* | 28.1 ms/doc |
| `centroid` | $k=20$ | 39.6 s | **79.6 %** *(+19.4 pt)* | 26.7 ms/doc |
| **`linear`** | **$k=20$** | 132 s | **85.2 %** *(+25.0 pt!)* | 41.9 ms/doc |

*Key takeaway:* `centroid` delivers massive gains (64.8 % → 79.6 %) with **zero training overhead** (compile is a pure embedding pass), while `linear` reaches **85.2 %** with 20 examples per class.

> **Anchor Rule of Thumb:** While 2–3 example sentences per class work as a quick zero-shot baseline (60–65 % on 77 classes), production schemas benefit significantly from providing **10–20 representative sentences** per category. Combined with `classifier="linear"` or `centroid`, this pushes accuracy into the **80–85 %+** range while keeping evaluation in the ~25–40 ms range on CPU.

### Throughput & Efficiency

* **Single Interactive Call:** **~46 ms** (p50) on dev CPU.
* **Bulk Processing (`decide_batch`):** **~13.9 ms/doc (72 docs/s)** sustained $\to$ **100,000 documents in ~24 minutes** on 1 core.
* **vs. Local LLMs (Ollama `qwen3.5:2b` / `nimble:9b`):** A 2B LLM on the same CPU achieves 40.5 % (+4.3 pt over Klix k=3) on MASSIVE-de, but takes **~9,000 ms/call (650× slower)** and requires **2.7 GB to 9.5 GB** memory. Klix provides the instant, deterministic System-1 layer.

---

## Choosing a Classifier Mode (3 Simple Rules)

1. **`classifier="centroid"` — Recommended for 90 % of schemas:**
   Scores against the normalized mean anchor vector per category. Fully deterministic, zero training overhead, robust against outliers, and scales independently of anchor count.
2. **`classifier="linear"` — For maximum accuracy on single-language schemas with 5+ anchors:**
   Fits a regularized logistic decision boundary at compile time. Reaches 85.2 % on 77 classes.
3. **`classifier="hybrid"` — When matching technical codes, asset IDs, or SKUs:**
   Fuses dense semantics with exact BM25 keyword matching (e.g. matching `plc-34` or error codes).

---

## Configuration Reference

| Parameter | Applies To | Description |
|---|---|---|
| `options` | `Choice`, `MultiLabel` | Mapping of `{label: [example_sentences, ...]}` |
| `classifier` | `Choice`, `MultiLabel` | `"centroid"` (default), `"linear"`, `"nearest"`, `"hybrid"` |
| `threshold` | `MultiLabel`, `Flag` | Decision threshold for active categories / boolean flag |
| `sharpness` / `center` | `MultiLabel` | Sigmoid steepness and cosine similarity center for $[0, 1]$ scoring |
| `calibration` | `MultiLabel` | Score mapping: `"sigmoid"` (default), `"linear"`, `"cosine"` |
| `reject_anchors` | `Choice` | Out-of-domain examples; matching queries return `value=None` |
| `neutral_anchors` | `Flag` | Third pole for ambiguous/irrelevant queries (returns `value=None`) |
| `sparse_fastpath` | `DecisionEngine` | Opt-in early-exit gate: resolves exact keyword matches in < 1 ms before dense pass |
| `glossary` | `Choice`, `DecisionEngine` | Language-agnostic concept map bridging synonyms cross-lingually |
| `truncate_dim` | `DecisionEngine` | Matryoshka dimension truncation (e.g. 128 dims) for lower memory & latency |

---

## Batch Mode & Multiprocessing

Process large document streams in a single embedding pass:

```python
texts = ["First document text...", "Second document text...", ...]
results = engine.decide_batch(texts)  # 72 docs/s on CPU
```

---

## License

MIT License. Designed and built for fast, local, reproducible text intelligence.
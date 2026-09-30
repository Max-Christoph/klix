# Comprehensive Benchmark Reports & Historical Sweeps

This document preserves the extended comparative benchmarks, ablations, and historical evaluations for `klix-engine`. Every figure here is reproducible via the scripts in `evals/`.

---

## 1. Large-Scale Public Datasets (BANKing77, MASSIVE)

Both datasets represent repurposed evaluation runs: Klix expects example anchor sentences per class, whereas these datasets ship only `text` and `label`. Anchors are generated via `evals/bespoke_anchors.py`.

* **`few_shot_k3`**: $k=3$ real sentences drawn from the training split with zero leakage (`assert_no_leakage()`).
* **`label_string`**: The bare label string with underscores replaced by spaces (honest lower bound).

| Dataset | Classes | Test cases | Anchors / Method | Accuracy | 95% CI |
|---|:---:|:---:|---|:---:|:---:|
| **BANKing77** | 77 | 3,080 | few_shot_k3 | **61.4 %** | [59.8, 63.2] |
| BANKing77 | 77 | 3,080 | label_string | 53.3 % | [51.5, 55.1] |
| **MASSIVE (en)** | 60 | 2,974 | few_shot_k3 | **43.8 %** | [41.9, 45.7] |
| MASSIVE (en) | 60 | 2,974 | label_string | **47.9 %** | [46.1, 49.7] |
| **MASSIVE (de)** | 60 | 2,974 | few_shot_k3 | **36.2 %** | [34.5, 38.0] |
| MASSIVE (de) | 60 | 2,974 | label_string | 30.1 % | [28.5, 31.8] |

### Known Dataset Flaw in MASSIVE
MASSIVE's `train` and `test` splits are not sentence-disjoint. In German, 115/2,974 (3.9%) of test sentences appear verbatim in `train`, **8 of them with contradictory labels** (identical sentence, two different intents). In English, 21/2,974 (0.7%) appear in train, 2 contradictory. This caps achievable German accuracy and is explicitly guarded by unit tests. Details in `evals/data/bespoke/PROVENANCE.md`.

---

## 2. Accuracy Scaling with Anchor Density (BANKing77, n=500 subsample)

On 77 fine-grained classes (random chance = 1.3%), scaling the anchor set from 3 to 20 examples per class dramatically lifts accuracy without changing the inference footprint. Evaluated on a deterministic 500-case subsample (full split baseline $n=3,080$ scores 61.4% for $k=3$ nearest, see Section 1):

| Classifier | Anchors / class ($k$) | Compile Time | Accuracy | Bulk Latency |
|---|:---:|:---:|:---:|:---:|
| `nearest` | $k=3$ | 4.7 s | 60.2 % | 29.7 ms/doc |
| `centroid` | $k=3$ | 5.3 s | **64.8 %** (+4.6 pt) | 26.6 ms/doc |
| `linear` | $k=3$ | 10.5 s | **67.8 %** (+7.6 pt) | 26.1 ms/doc |
| `centroid` | $k=10$ | 22.4 s | **76.6 %** (+16.4 pt) | 31.3 ms/doc |
| `linear` | $k=10$ | 85.0 s | **79.8 %** (+19.6 pt) | 28.1 ms/doc |
| `centroid` | $k=20$ | 39.6 s | **79.6 %** (+19.4 pt) | 26.7 ms/doc |
| `linear` | **$k=20$** | 132 s | **85.2 %** (+25.0 pt) | 41.9 ms/doc |

*Evaluated on 500 subsampled test cases from BANKing77. All measurements are backed by committed evaluation runs in `evals/`: `bespoke_result_banking77_few_shot_k3_nearest.json`, `..._k3_centroid.json`, and `..._k20_linear.json`.*

---

## 3. Multi-Label Classification (GoEmotions, 28 Categories)

Evaluates the `MultiLabel` head on Google Research GoEmotions (Reddit comments labeled with 27 emotions + neutral; multi-label distribution where samples can carry multiple emotions simultaneously).

* **Architecture:** Centroid aggregation (`_centroid_matrix @ q`), sigmoid calibration (`sharpness=12.0`, `center=0.42`, `threshold=0.60`).
* **Anchors:** Monolabel-prioritized few-shot anchors drawn from `train` only. Leakage check: `assert_no_leakage()` confirmed **0 test leaks**.
* **Footprint:** Single matrix-vector dot product evaluating 28 categories in **< 5 µs** per query after backbone embedding.

| Setup | Anchors / class ($k$) | Micro-F1 | Macro-F1 | Precision | Recall | Bulk Throughput | Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `MultiLabel` (centroid) | $k=3$ | 9.81 % | 7.13 % | 15.89 % | 7.09 % | 51.8 docs/s | 19.3 ms/doc |
| `MultiLabel` (centroid) | $k=10$ | **19.51 %** *(+9.7 pt)* | **18.10 %** | **17.03 %** | **22.84 %** | **55.0 docs/s** | **18.2 ms/doc** |

*Evaluated on 500 subsampled test cases from GoEmotions. Fully reproducible via `evals/run_multilabel.py` with committed result JSONs in `evals/bespoke_result_go_emotions_k3.json` and `..._k10.json`.*

---

## 4. vs. Local 2B/9B LLMs (Ollama)

Evaluated on 500 MASSIVE-de cases on an Intel Core Ultra 5 CPU (no dedicated GPU):

| Metrik | Klix (`few_shot_k3`) | `qwen3.5:2b` (Ollama) | Nimble 9B (Est. CPU) |
|---|---|---|---|
| **Accuracy** | 36.2 % (full split) | 40.5 % (500 subsample) | ~50–60 % |
| **Model Size** | **240 MB** | 2.7 GB | 9.5 GB |
| **Sustained Latency** | **13.9 ms / doc** | ~9,000 ms / call | ~20,000 ms / call |
| **Throughput** | **72 docs/s** | ~0.11 docs/s | ~0.05 docs/s |
| **100k Documents** | **~24 minutes** | ~10 days | ~23–30 days |
| **Coverage** | 100 % | 99.2 % (4 unparseable) | 99 % |

The LLM buys +4.3 percentage points on German MASSIVE at the cost of **650× higher latency** and **11× larger footprint**.

---

## 5. Cross-Domain Routing (6 domains, 70 cases)

Evaluated via `evals/benchmark.py` (HR, Finance, Image-captions, Tasks, Shop, Guardrails):

| Method | Avg Accuracy | Median Latency |
|---|:---:|:---:|
| TF-IDF + LogReg | 51 % | ~1–5 ms |
| Embed-KNN (dense) | 78 % | ~58 ms |
| Klix `nearest` | 72 % | ~71 ms |
| **Klix `linear`** | **84 %** | ~80 ms |
| **Klix `centroid`** | **84.3 %** | ~72 ms |

---

## 6. vs. SetFit (Contrastive Few-Shot Training)

`evals/setfit_baseline.py` — SetFit trained on the same anchor texts (`num_epochs=1`, MiniLM backbone, CPU):

| Dataset | Klix `nearest` | Klix `linear` | SetFit (Trained) |
|---|:---:|:---:|:---:|
| HR | 8/12 | 9/12 | 9/12 |
| FIN | 10/12 | 12/12 | 11/12 |
| IMAGE | 9/12 | 11/12 | 11/12 |
| TASK | 7/12 | 11/12 | 10/12 |
| SHOP | 7/12 | 8/12 | 10/12 |
| **Total (n=60)** | 41/60 (68 %) | **51/60 (85 %)** | **51/60 (85 %)** |

**Finding:** Klix `linear` matches SetFit few-shot training accuracy (85%) with zero gradient updates and instant compilation.

---

## 7. vs. Laya (`convaiinnovations/laya`)

Measured on the 70 cross-domain cases on **CPU**:

| Metrik | Klix `linear` | Laya (Zero-Shot) |
|---|:---:|:---:|
| **Accuracy** | **86 %** | 67 % |
| **Latency (CPU)** | **~13 ms** | ~1.3–1.5 s |
| **Model Size** | **240 MB** | ~2 GB |
| **Setup** | Anchors (Few-shot) | Instructions + Criteria (Zero-shot) |

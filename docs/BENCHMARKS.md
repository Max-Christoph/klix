# Comprehensive Benchmark Reports & Historical Sweeps

This document preserves the extended comparative benchmarks, ablations, and historical evaluations for `klix-engine`. Every figure here is reproducible via the scripts in `evals/`.

---

## 1. Large-Scale Public Datasets (BANKing77, MASSIVE)

Both datasets represent repurposed evaluation runs: Klix expects example anchor sentences per class, whereas these datasets ship only `text` and `label`. Anchors are generated via `evals/bespoke_anchors.py`.

* **`few_shot_k3`**: $k=3$ real sentences drawn from the training split with zero leakage (`assert_no_leakage()`).
* **`label_string`**: The bare label string with underscores replaced by spaces (honest lower bound).
* **`few_shot_k3_plus_label`**: Combines the canonical label string with $k=3$ real training examples under centroid aggregation.

| Dataset | Classes | Test cases | Anchors / Method | Accuracy | 95% CI |
|---|:---:|:---:|---|:---:|:---:|
| **BANKing77** | 77 | 3,080 | few_shot_k3 | **61.4 %** | [59.8, 63.2] |
| BANKing77 | 77 | 3,080 | label_string | 53.3 % | [51.5, 55.1] |
| **MASSIVE (en)** | 60 | 2,974 | few_shot_k3 | **43.8 %** | [41.9, 45.7] |
| MASSIVE (en) | 60 | 2,974 | label_string | **47.9 %** | [46.1, 49.7] |
| **MASSIVE (en)** | 60 | 2,974 | **few_shot_k3_plus_label (centroid)** | **57.3 %** | [55.5, 59.1] |
| **MASSIVE (de)** | 60 | 2,974 | few_shot_k3 | **36.2 %** | [34.5, 38.0] |
| MASSIVE (de) | 60 | 2,974 | label_string | 30.1 % | [28.5, 31.8] |
| **MASSIVE (de)** | 60 | 2,974 | **few_shot_k3_plus_label (centroid)** | **44.4 %** | [42.6, 46.2] |

*Ablation Finding:* On MASSIVE, adding the canonical label name to the few-shot anchor set and applying centroid aggregation lifts accuracy by **+13.5 percentage points** on English (57.3% vs. 43.8%) and **+8.2 percentage points** on German (44.4% vs. 36.2%), outperforming the local Qwen-2B autoregressive LLM baseline (40.5%).

### Known Dataset Flaw in MASSIVE
MASSIVE's `train` and `test` splits are not sentence-disjoint. In German, 115/2,974 (3.9%) of test sentences appear verbatim in `train`, **8 of them with contradictory labels** (identical sentence, two different intents). In English, 21/2,974 (0.7%) appear in train, 2 contradictory. This caps achievable German accuracy and is explicitly guarded by unit tests. Details in `evals/data/bespoke/PROVENANCE.md`.

---

## 2. Out-of-Scope (OOD) Detection & Intent Routing (CLINC150, 150 Classes)

Evaluates in-scope intent routing across 150 fine-grained intent categories simultaneously with out-of-domain rejection on 1,000 out-of-scope (OOS) queries (Larson et al., EMNLP 2019). Provenance and attribution details in `evals/data/clinc/PROVENANCE.md`.

* **Data Split:** Full test split ($N=4,500$ in-scope test queries across 150 classes + $N=1,000$ out-of-scope queries = 5,500 total).
* **Prior Baseline:** Uniform random chance for in-scope routing is $1/150 = 0.67\%$.
* **OOD Scoring Metric:** Maximum class centroid cosine similarity ($\max_k (\mathbf{q} \cdot \mathbf{c}_k)$).

| Anchor Density ($k$) | Classifier | In-Scope Accuracy ($N=4,500$) | OOD AUROC ($N=5,500$) | OOD FPR@95 | Latency | Sustained Throughput |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $k=3$ anchors/class | `centroid` | **78.78 %** | **92.92 %** | **32.30 %** | 10.54 ms/doc | **94.9 docs/s** |
| $k=10$ anchors/class | `centroid` | **86.93 %** | **94.79 %** | **23.10 %** | 10.09 ms/doc | **99.1 docs/s** |

*Key Findings:*
1. **OOD Discrimination:** Maximum centroid cosine similarity achieves **94.79% AUROC** on OOD detection with $k=10$ anchors per category without any external classifier overhead.
2. **False Positive Suppression:** At a 95% True Positive Rate (accepting 95% of legitimate in-scope requests), the False Positive Rate on unseen out-of-scope queries drops to **23.10%**.
3. **Execution Speed:** Evaluating 150 intent classes and computing OOD rejection executes in **~10.1 ms per query** on standard CPU hardware (sustained 99.1 docs/s). Fully reproducible via `evals/run_clinc_oos.py`.

---

## 3. Accuracy Scaling with Anchor Density (BANKing77, n=500 subsample)

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

## 4. Comparison with Autoregressive Small Language Models (Ollama)

Evaluated on 500 MASSIVE-de cases on an Intel Core Ultra 5 CPU (no dedicated GPU):

| Metric | Klix (`few_shot_k3`) | `qwen3.5:2b` (Ollama) | Nimble 9B (Est. CPU) |
|---|---|---|---|
| **Accuracy** | 36.2 % (full split) | 40.5 % (500 subsample) | ~50–60 % |
| **Model Footprint** | **240 MB** | 2.7 GB | 9.5 GB |
| **Median Per-Query Latency** | **13.9 ms / doc** | ~9,000 ms / call | ~20,000 ms / call |
| **Throughput** | **72 docs/s** | ~0.11 docs/s | ~0.05 docs/s |
| **100k Documents (Est.)** | **~24 minutes** | ~10 days | ~23–30 days |
| **Output Validity** | 100 % (deterministic) | 99.2 % (4 unparseable) | 99 % |

*Analysis:* On German MASSIVE, an autoregressive 2B parameter model achieves 40.5% accuracy (+4.3 percentage points over Klix $k=3$), at the trade-off of ~9,000 ms median latency and 2.7 GB memory allocation. Klix provides bounded linear-algebra latency (13.9 ms/doc) suitable for real-time routing gates.

---

## 5. Cross-Domain Routing Sanity Suite (6 Domains, N=70)

Small-scale sanity test suite ($N=70$ cases across 6 distinct domain schemas: HR, Finance, Image-captions, Tasks, Shop, Guardrails) designed to evaluate cross-domain schema stability via `evals/benchmark.py`:

| Method | Mean Accuracy ($N=70$) | Median Latency |
|---|:---:|:---:|
| TF-IDF + Logistic Regression | 51.4 % | ~1–5 ms |
| Embed-KNN (dense anchor matching) | 78.0 % | ~58 ms |
| Klix `nearest` | 72.9 % | ~71 ms |
| Klix `linear` | 84.3 % | ~80 ms |
| **Klix `centroid`** | **84.3 %** | ~72 ms |

*Note:* This test suite serves as a quick sanity check for schema composition. For high-cardinality statistical characterization, refer to the full BANKing77 ($N=3,080$) and MASSIVE ($N=2,974$) test splits.

---

## 6. Contrastive Fine-Tuning Comparison (SetFit, N=60)

`evals/setfit_baseline.py` — Evaluates whether an $L_2$-regularized linear probe over frozen embeddings performs comparably to a contrastively fine-tuned model (`SetFit`, `num_epochs=1`, MiniLM backbone, CPU) on the identical anchor set:

| Domain | Klix `nearest` | Klix `linear` | SetFit (Fine-Tuned) |
|---|:---:|:---:|:---:|
| HR | 8/12 | 9/12 | 9/12 |
| FIN | 10/12 | 12/12 | 11/12 |
| IMAGE | 9/12 | 11/12 | 11/12 |
| TASK | 7/12 | 11/12 | 10/12 |
| SHOP | 7/12 | 8/12 | 10/12 |
| **Total (N=60)** | 41/60 (68.3 %) | **51/60 (85.0 %)** | **51/60 (85.0 %)** |

*Observation ($N=60$):* On this 5-domain sample, fitting an $L_2$-regularized convex linear probe over frozen embeddings attains 85.0% accuracy, matching the contrastively fine-tuned model without requiring backpropagation passes or PyTorch runtime dependencies.

---

## 7. Comparison with Zero-Shot NLI / Instruction Models (Laya, N=70)

Measured on the 70 cross-domain cases on CPU to compare anchor-based vector projection against zero-shot natural language inference parsing:

| Metric | Klix `linear` (Anchors) | Laya (Zero-Shot Instructions) |
|---|:---:|:---:|
| **Accuracy ($N=70$)** | **85.7 %** | 67.1 % |
| **Latency (CPU)** | **~13 ms** | ~1,300–1,500 ms |
| **Model Size** | **240 MB** | ~2 GB |
| **Configuration** | Anchor examples ($k \ge 3$) | Natural language instructions + criteria |


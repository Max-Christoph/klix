"""Benchmark: klix vs. standard baselines on identical anchor data.

The fair comparison: every method is "trained" on the SAME anchors (the few-shot
example sentences). This isolates what klix actually adds over the obvious
scikit-learn / embedding baselines.

Methods, ordered by complexity:
  1. tfidf+lr        - pure TF-IDF + LogisticRegression (no embeddings)
  2. embed-knn       - dense MiniLM embedding + nearest-anchor (klix minus sparse)
  3. klix-nearest    - dense + sparse hybrid (keyword boost)
  4. klix-linear     - dense + learned linear probe

Metrics per dataset: in-domain accuracy, plus per-query inference latency
(embedding-based methods include the ~10ms MiniLM forward pass).

Run: uv run python -c "import sys; sys.path.insert(0,'.'); sys.path.insert(0,'src'); import evals.benchmark"
"""

import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402
from fastembed import TextEmbedding  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from klix import DecisionEngine, Choice  # noqa: E402
from evals.eval_domains import (  # noqa: E402
    IMG_OPTIONS, TASK_OPTIONS, SHOP_OPTIONS, IMG_CASES, TASK_CASES, SHOP_CASES,
)
from evals.variant_sweep import HR_OPTIONS, FIN_OPTIONS, HR_TESTS, FIN_TESTS  # noqa: E402
from evals.eval_guardrail import OPTIONS as GUARD_OPTIONS, TESTS as GUARD_TESTS  # noqa: E402

# ---------------------------------------------------------------------------
# Datasets: (name, options, in-domain labeled test cases)
# ---------------------------------------------------------------------------
DATASETS = [
    ("HR", HR_OPTIONS, HR_TESTS),
    ("FIN", FIN_OPTIONS, FIN_TESTS),
    ("IMAGE", IMG_OPTIONS, [(q, e) for q, e in IMG_CASES if e]),
    ("TASK", TASK_OPTIONS, [(q, e) for q, e in TASK_CASES if e]),
    ("SHOP", SHOP_OPTIONS, [(q, e) for q, e in SHOP_CASES if e]),
    ("GUARD", GUARD_OPTIONS, [(q, e) for q, e in GUARD_TESTS if e]),
]


def _flatten(options: dict) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    for label, examples in options.items():
        for ex in examples:
            texts.append(ex)
            labels.append(label)
    return texts, labels


# ---------------------------------------------------------------------------
# Method 1: TF-IDF + LogisticRegression
# ---------------------------------------------------------------------------
def run_tfidf_lr(options: dict, tests: list, embed) -> tuple[int, int, list[float]]:
    texts, labels = _flatten(options)
    vec = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b[\w-]+\b", lowercase=True)
    vec.fit(texts)
    X = vec.transform(texts)
    clf = LogisticRegression(C=10.0, max_iter=1000, class_weight="balanced")
    clf.fit(X, labels)
    ok = 0
    times = []
    for text, expected in tests:
        t0 = time.perf_counter()
        pred = clf.predict(vec.transform([text]))[0]
        times.append((time.perf_counter() - t0) * 1000)
        if pred == expected:
            ok += 1
    return ok, len(tests), times


# ---------------------------------------------------------------------------
# Method 2: Embedding KNN (dense only, nearest anchor)
# ---------------------------------------------------------------------------
def run_embed_knn(options: dict, tests: list, embed) -> tuple[int, int, list[float]]:
    texts, labels = _flatten(options)
    ref = np.array(list(embed.embed(texts)))
    ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    ok = 0
    times = []
    for text, expected in tests:
        t0 = time.perf_counter()
        q = np.array(list(embed.embed([text]))[0])
        q = q / (np.linalg.norm(q) or 1.0)
        sims = ref @ q
        pred = labels[int(np.argmax(sims))]
        times.append((time.perf_counter() - t0) * 1000)
        if pred == expected:
            ok += 1
    return ok, len(tests), times


# ---------------------------------------------------------------------------
# Method 3 & 4: klix nearest / linear
# ---------------------------------------------------------------------------
def run_klix(options: dict, tests: list, classifier: str) -> tuple[int, int, list[float]]:
    eng = DecisionEngine()
    eng.add_head(Choice(name="c", options=options, classifier=classifier))
    eng.compile()
    ok = 0
    times = []
    for text, expected in tests:
        t0 = time.perf_counter()
        got = eng.decide(text).c
        times.append((time.perf_counter() - t0) * 1000)
        if got == expected:
            ok += 1
    return ok, len(tests), times


def main():
    # Load the embedding model ONCE for all embedding-based baselines.
    print("Loading embedding model (once)...")
    embed = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    methods = ["tfidf+lr", "embed-knn", "klix-nearest", "klix-linear"]
    results = {m: [] for m in methods}
    latencies = {m: [] for m in methods}

    for name, options, tests in DATASETS:
        row_ok = {}
        row_lat = {}
        for m in methods:
            if m == "tfidf+lr":
                ok, n, times = run_tfidf_lr(options, tests, embed)
            elif m == "embed-knn":
                ok, n, times = run_embed_knn(options, tests, embed)
            elif m == "klix-nearest":
                ok, n, times = run_klix(options, tests, "nearest")
            else:
                ok, n, times = run_klix(options, tests, "linear")
            row_ok[m] = f"{ok}/{n}"
            row_lat[m] = np.median(times)
            results[m].append(ok / n)
            latencies[m].append(np.median(times))
        print(f"{name:8s} " + "  ".join(f"{m}:{row_ok[m]}" for m in methods))

    # Summary
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'method':14s} {'avg acc':>9s} {'median lat/query':>18s}")
    print("-" * 78)
    for m in methods:
        avg_acc = np.mean(results[m])
        avg_lat = np.mean(latencies[m])
        print(f"{m:14s} {avg_acc:>8.0%} {avg_lat:>15.1f} ms")
    print("-" * 78)
    print("Note: embed-knn / klix-* include the embedding forward pass,")
    print("which dominates latency (~50-90 ms per query on the dev")
    print("workstation, 2026-09-24; hardware-dependent).")
    print("tfidf+lr does not (no embeddings) — that's its speed advantage.")

    # Setup effort (qualitative)
    print("\nSetup effort (qualitative):")
    print("  tfidf+lr      : ~6 lines sklearn, no model download")
    print("  embed-knn     : ~8 lines + ~120MB model download (once)")
    print("  klix-*        : declarative schema + compile(); handles reject/coverage/confidence")


if __name__ == "__main__":
    main()

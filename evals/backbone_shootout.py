"""Backbone shootout: compare alternative FastEmbed embedding models on CPU.

Evaluates candidates on:
  1. Cross-Domain benchmark (6 domains, 70 cases from evals.benchmark)
     - nearest and linear accuracy
  2. Bilingual benchmark (20 parallel cases: 10 EN, 10 DE from evals.benchmark_bilingual)
     - nearest and linear accuracy (split EN / DE)
  3. Latency on CPU (50 iterations):
     - median and p95 latency in milliseconds
  4. Memory / model size / dimensions

Run:
    uv run python evals/backbone_shootout.py
"""
from __future__ import annotations

import statistics
import sys
import time
import warnings

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from klix import Choice, DecisionEngine  # noqa: E402
from evals.benchmark import DATASETS as CROSS_DOMAIN_DATASETS  # noqa: E402
from evals.benchmark_bilingual import OPTIONS as BILINGUAL_OPTIONS, TEST_EN, TEST_DE  # noqa: E402

MODELS = [
    {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "label": "MiniLM-L12-multi (Default)",
        "dim": 384,
        "size_mb": 118,
        "lang": "Multilingual (50+)",
    },
    {
        "name": "BAAI/bge-small-en-v1.5",
        "label": "BGE-small-en-v1.5",
        "dim": 384,
        "size_mb": 67,
        "lang": "English",
    },
    {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "label": "All-MiniLM-L6-v2",
        "dim": 384,
        "size_mb": 90,
        "lang": "English",
    },
    {
        "name": "snowflake/snowflake-arctic-embed-xs",
        "label": "Arctic-embed-xs",
        "dim": 384,
        "size_mb": 90,
        "lang": "English",
    },
    {
        "name": "jinaai/jina-embeddings-v2-base-de",
        "label": "Jina-v2-base-de",
        "dim": 768,
        "size_mb": 320,
        "lang": "DE + EN",
    },
]


def eval_cross_domain(model_name: str, classifier: str) -> tuple[int, int]:
    total_ok = 0
    total_cases = 0
    for name, options, tests in CROSS_DOMAIN_DATASETS:
        engine = DecisionEngine(model_name=model_name)
        engine.add_head(Choice(name=name.lower(), options=options, classifier=classifier))
        engine.compile()
        for text, expected in tests:
            total_cases += 1
            res = engine.decide(text)
            if getattr(res, name.lower()) == expected:
                total_ok += 1
    return total_ok, total_cases


def eval_bilingual(model_name: str, classifier: str) -> tuple[int, int, int, int]:
    engine = DecisionEngine(model_name=model_name)
    engine.add_head(Choice(name="target", options=BILINGUAL_OPTIONS, classifier=classifier))
    engine.compile()

    en_ok = sum(1 for text, exp in TEST_EN if engine.decide(text).target == exp)
    de_ok = sum(1 for text, exp in TEST_DE if engine.decide(text).target == exp)
    return en_ok, len(TEST_EN), de_ok, len(TEST_DE)


def measure_latency(model_name: str, n_samples: int = 50) -> tuple[float, float]:
    engine = DecisionEngine(model_name=model_name)
    engine.add_head(Choice(name="test", options=BILINGUAL_OPTIONS, classifier="nearest"))
    engine.compile()

    # warmup
    for _ in range(5):
        engine.decide("sensor calibration overdue")

    latencies = []
    for _ in range(n_samples):
        t0 = time.perf_counter()
        engine.decide("sensor calibration overdue")
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    median = statistics.median(latencies)
    p95 = latencies[int(0.95 * (len(latencies) - 1))]
    return median, p95


def main():
    warnings.filterwarnings("ignore")
    print("=" * 80)
    print("KLIX BACKBONE SHOOTOUT (CPU)")
    print("=" * 80)
    print()

    results = []
    for m in MODELS:
        name = m["name"]
        print(f"Evaluating {m['label']} ...")
        
        # 1. Latency
        med_ms, p95_ms = measure_latency(name)
        print(f"  Latency: median={med_ms:.1f}ms, p95={p95_ms:.1f}ms")

        # 2. Cross Domain
        cd_near_ok, cd_total = eval_cross_domain(name, "nearest")
        cd_lin_ok, _ = eval_cross_domain(name, "linear")
        print(f"  Cross-Domain (n={cd_total}): nearest={cd_near_ok}/{cd_total} ({cd_near_ok/cd_total:.1%}), linear={cd_lin_ok}/{cd_total} ({cd_lin_ok/cd_total:.1%})")

        # 3. Bilingual
        bi_en_near, bi_en_tot, bi_de_near, bi_de_tot = eval_bilingual(name, "nearest")
        bi_near_tot = bi_en_near + bi_de_near
        bi_tot = bi_en_tot + bi_de_tot

        bi_en_lin, _, bi_de_lin, _ = eval_bilingual(name, "linear")
        bi_lin_tot = bi_en_lin + bi_de_lin
        print(f"  Bilingual (n={bi_tot}): nearest={bi_near_tot}/{bi_tot} (EN {bi_en_near}/{bi_en_tot}, DE {bi_de_near}/{bi_de_tot}), linear={bi_lin_tot}/{bi_tot} (EN {bi_en_lin}/{bi_en_tot}, DE {bi_de_lin}/{bi_de_tot})")
        print()

        results.append({
            "meta": m,
            "latency_med": med_ms,
            "latency_p95": p95_ms,
            "cd_near": cd_near_ok / cd_total,
            "cd_lin": cd_lin_ok / cd_total,
            "bi_near": bi_near_tot / bi_tot,
            "bi_near_en": bi_en_near,
            "bi_near_de": bi_de_near,
            "bi_lin": bi_lin_tot / bi_tot,
            "bi_lin_en": bi_en_lin,
            "bi_lin_de": bi_de_lin,
        })

    print("=" * 80)
    print("FINAL SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Model':<30} | {'Size':<6} | {'Dim':<4} | {'Med ms':<6} | {'p95 ms':<6} | {'CD Near':<7} | {'CD Lin':<7} | {'Bi Near':<7} | {'Bi Lin':<7}"
    print(header)
    print("-" * len(header))
    for r in results:
        m = r["meta"]
        print(
            f"{m['label']:<30} | "
            f"{m['size_mb']}MB   | "
            f"{m['dim']:<4} | "
            f"{r['latency_med']:>5.1f}  | "
            f"{r['latency_p95']:>5.1f}  | "
            f"{r['cd_near']:>6.1%}  | "
            f"{r['cd_lin']:>6.1%}  | "
            f"{r['bi_near']:>6.1%}  | "
            f"{r['bi_lin']:>6.1%}"
        )


if __name__ == "__main__":
    main()

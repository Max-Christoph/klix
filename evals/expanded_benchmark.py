"""E.1 expanded benchmark: paraphrase corpus + bootstrap 95% CIs.

Measures nearest / linear / bm25+topk3+coverage on the expanded corpus
(frozen 60 + generated variants) with deterministic bootstrap confidence
intervals. Read the absolute numbers as INFLATED (see corpus_expansion
docstring); use the CIs to compare configs against each other.
"""
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np

from klix import DecisionEngine, Choice
from evals.corpus_expansion import build_expanded_corpus
from evals.variant_sweep import HR_OPTIONS, FIN_OPTIONS, HR_TESTS, FIN_TESTS
from evals.eval_domains import (
    IMG_OPTIONS, TASK_OPTIONS, SHOP_OPTIONS, IMG_CASES, TASK_CASES, SHOP_CASES,
)

SETS = [
    ("HR", HR_OPTIONS, HR_TESTS),
    ("FIN", FIN_OPTIONS, FIN_TESTS),
    ("IMAGE", IMG_OPTIONS, [(q, e) for q, e in IMG_CASES if e]),
    ("TASK", TASK_OPTIONS, [(q, e) for q, e in TASK_CASES if e]),
    ("SHOP", SHOP_OPTIONS, [(q, e) for q, e in SHOP_CASES if e]),
]

CONFIGS = [
    ("nearest (status quo)", {}),
    ("linear (dense-only)", {"classifier": "linear"}),
    ("bm25+topk3+cov", {"sparse_metric": "bm25", "label_aggregation": "topk",
                        "label_topk": 3, "keyword_boost_mode": "coverage"}),
]


def ci95(correct: list[bool], n_boot: int = 300, seed: int = 42) -> tuple[float, float]:
    """Deterministic bootstrap 95% CI for an accuracy point estimate."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(correct, dtype=float)
    n = len(arr)
    if n < 4:
        return (0.0, 1.0)
    metrics = sorted(arr[rng.integers(0, n, size=n)].mean() for _ in range(n_boot))
    return (round(metrics[int(0.025 * len(metrics))], 4),
            round(metrics[min(len(metrics) - 1, int(0.975 * len(metrics)))], 4))


def main() -> None:
    corpus = build_expanded_corpus(SETS)

    print("=" * 104)
    print("EXPANDED BENCHMARK (paraphrase corpus) — configs with bootstrap 95% CI")
    print("=" * 104)
    print(f"{'CONFIG':22s}" + "".join(f"{n:>10s}" for n, _, _ in SETS)
          + f"{'TOTAL':>11s} {'ci95':>16s} {'ms/case':>9s}")
    print("-" * 104)

    for vname, kw in CONFIGS:
        per_domain: dict[str, tuple[int, int]] = {}
        all_correct: list[bool] = []
        total_ms = 0.0
        for name, options, _tests in SETS:
            eng = DecisionEngine()
            eng.add_head(Choice(name="h", options=options, **kw))
            eng.compile()
            ok = 0
            for ticket, expected in corpus["domains"][name]["cases"]:
                t0 = time.perf_counter()
                got = eng.decide(ticket).h
                total_ms += (time.perf_counter() - t0) * 1000
                if got == expected:
                    ok += 1
                    all_correct.append(True)
                else:
                    all_correct.append(False)
            per_domain[name] = (ok, len(corpus["domains"][name]["cases"]))
        total = sum(o for o, _ in per_domain.values())
        grand = sum(nn for _, nn in per_domain.values())
        lo, hi = ci95(all_correct)
        print(f"{vname:22s}"
              + "".join(f"{o:>5d}/{nn:<4d}" for o, nn in per_domain.values())
              + f" {total:>4d}/{grand:<5d} {lo:.1%}-{hi:.0%} {total_ms/grand:>9.1f}")

    print("-" * 104)
    print(f"Corpus: {sum(len(d['cases']) for d in corpus['domains'].values())} cases "
          f"(frozen 60 + {corpus['total'] - 60} generated).")
    print("Honesty: paraphrase accuracy is inflated vs the frozen 60; compare configs, not absolutes.")

    # --- Frozen 60 under the same configs (the honest absolute baseline) ---
    print()
    print("=" * 104)
    print("FROZEN 60 BASELINE — same configs on the untouched labeled cases")
    print("=" * 104)
    print(f"{'CONFIG':22s}" + "".join(f"{n:>10s}" for n, _, _ in SETS)
          + f"{'TOTAL':>11s} {'ci95':>16s}")
    print("-" * 104)
    for vname, kw in CONFIGS:
        per_domain: dict[str, tuple[int, int]] = {}
        all_correct: list[bool] = []
        for name, options, tests in SETS:
            eng = DecisionEngine()
            eng.add_head(Choice(name="h", options=options, **kw))
            eng.compile()
            ok = 0
            for ticket, expected in tests:
                if eng.decide(ticket).h == expected:
                    ok += 1
                    all_correct.append(True)
                else:
                    all_correct.append(False)
            per_domain[name] = (ok, len(tests))
        total = sum(o for o, _ in per_domain.values())
        grand = sum(nn for _, nn in per_domain.values())
        lo, hi = ci95(all_correct)
        print(f"{vname:22s}"
              + "".join(f"{o:>5d}/{nn:<4d}" for o, nn in per_domain.values())
              + f" {total:>4d}/{grand:<5d} {lo:.1%}-{hi:.0%}")
    print("-" * 104)


if __name__ == "__main__":
    main()
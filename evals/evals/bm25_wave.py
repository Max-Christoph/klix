"""BM25 vs TF-IDF on the unchanged labeled sets (D.2 experiment).

Reuses the exact anchor sets and labeled cases from linear_sweep.
Compares: tfidf defaults (status quo), bm25, bm25+topk2+coverage.
"""
import time

from linear_sweep import (
    HR_OPTIONS, HR_TESTS, FIN_OPTIONS, FIN_TESTS,
    IMG_OPTIONS, IMG_CASES, TASK_OPTIONS, TASK_CASES,
    SHOP_OPTIONS, SHOP_CASES,
)

from klix import DecisionEngine, Choice


def measure(options: dict, tests: list, **kw) -> tuple[int, int]:
    eng = DecisionEngine()
    eng.add_head(Choice(name="h", options=options, **kw))
    eng.compile()
    hits = sum(1 for ticket, expected in tests
               if eng.decide(ticket).h == expected)
    return hits, len(tests)


SETS = [
    ("HR", HR_OPTIONS, HR_TESTS),
    ("FIN", FIN_OPTIONS, FIN_TESTS),
    ("IMAGE", IMG_OPTIONS, [(q, e) for q, e in IMG_CASES if e]),
    ("TASK", TASK_OPTIONS, [(q, e) for q, e in TASK_CASES if e]),
    ("SHOP", SHOP_OPTIONS, [(q, e) for q, e in SHOP_CASES if e]),
]

if __name__ == "__main__":
    print("=" * 70)
    print(f"{'CONFIG':30s}" + "".join(f"{n:>9s}" for n, _, _ in SETS))
    print("-" * 70)
    for vname, kw in CONFIGS:
        eng = DecisionEngine()
        results = []
        for name, options, tests in SETS:
            engine = DecisionEngine()
            engine.add_head(Choice(name="h", options=options, **kw))
            engine.compile()
            hits = sum(1 for ticket, expected in tests
                       if engine.decide(ticket).h == expected)
            results.append(f"{hits}/{len}")
        print(f"{vname:28s}" + "".join(results))
    print("-" * 70)
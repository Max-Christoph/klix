"""Experiment: combine the winning knobs — BM25 sparse channel + topk/coverage aggregation.

Question: does bm25 on top of topk3+coverage push past linear's 85%?
Must NOT change anchors/test cases.
"""
from linear_sweep import HR_OPTIONS, HR_TESTS, FIN_OPTIONS, FIN_TESTS, IMG_OPTIONS, IMG_CASES, TASK_OPTIONS, TASK_CASES, SHOP_OPTIONS, SHOP_CASES

from klix import DecisionEngine, Choice


def measure(options: dict, tests: list, **kw) -> tuple[int, int]:
    eng = DecisionEngine()
    eng.add_head(Choice(name="h", options=options, **kw))
    eng.compile()
    ok = 0
    for ticket, expected in tests:
        if eng.decide(ticket).h == expected:
            ok += 1
    return ok, len(tests)


sets = [
    ("HR", HR_OPTIONS, HR_TESTS),
    ("FIN", FIN_OPTIONS, FIN_TESTS),
    ("IMAGE", IMG_OPTIONS, [(q, e) for q, e in IMG_CASES if e]),
    ("TASK", TASK_OPTIONS, [(q, e) for q, e in TASK_CASES if e]),
    ("SHOP", SHOP_OPTIONS, [(q, e) for q, e in SHOP_CASES if e]),
]

variants = [
    ("tfidf + max + fixed (status quo)", {}),
    ("bm25 + max + fixed", {"sparse_metric": "bm25"}),
    ("tfidf + topk2 + coverage", {"label_aggregation": "topk", "keyword_boost_mode": "coverage"}),
    ("bm25 + topk2 + coverage", {"sparse_metric": "bm25", "label_aggregation": "topk", "keyword_boost_mode": "coverage"}),
    ("bm25 + topk3 + coverage", {"sparse_metric": "bm25", "label_aggregation": "topk", "label_topk": 3, "keyword_boost_mode": "coverage"}),
    ("linear (dense-only, reference)", {"classifier": "linear"}),
]

print("=" * 90)
print("KNOB COMBINATION SWEEP — nearest path with bm25/topk/coverage vs linear reference")
print("=" * 88)
header = f"{'VARIANT':36s}" + "".join(f"{n:>9s}" for n, _, _ in sets) + f"{'TOTAL':>10s}"
print(header)
print("-" * 88)
for vname, kw in variants:
    cells = []
    ok_sum, n_sum = 0, 0
    for name, options, tests in sets:
        ok, n = measure(options, tests, **kw)
        ok_sum += ok
        n_sum += n
        cells.append(f"{ok:>5d}/{n:<3d}")
    print(f"{vname:36s}" + "".join(cells) + f"{ok_sum:>5d}/{n_sum} {ok_sum/n_sum:>6.1%}")
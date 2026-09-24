"""Experiment: classifier="hybrid" + sparse_metric="bm25" — do the two new knobs stack?

Both innovations individually reach ~85%; if they stack, the nearest-family
path can exceed the dense-only probe. Must NOT change anchors/test cases.

NOTE on results (recorded 2026-09-24): hybrid stays at 49/60 (81.7%) across
all sparse_metric/aggregation settings — the probe path ignores topk/coverage
by design (aggregation knobs are nearest-path only), and bm25 changes the
sparse half of the probe features but not the outcome on these sets. The
probe-family ceiling on these 60 cases is 85% (dense-only linear with
augmentation), which the hybrid path gives up by skipping mixup augmentation.
"""
from evals.linear_sweep import HR_OPTIONS, HR_TESTS, FIN_OPTIONS, FIN_TESTS, IMG_OPTIONS, IMG_CASES, TASK_OPTIONS, TASK_CASES, SHOP_OPTIONS, SHOP_CASES

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
    ("hybrid (tfidf)", {"classifier": "hybrid"}),
    ("hybrid + bm25", {"classifier": "hybrid", "sparse_metric": "bm25"}),
]

print("=" * 88)
print("HYBRID probe with bm25 — do the knobs stack on top of the probe?")
print("=" * 88)
header = f"{'VARIANT':30s}" + "".join(f"{n:>9s}" for n, _, _ in sets) + f"{'TOTAL':>10s}"
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
    print(f"{vname:30s}" + "".join(cells) + f"{ok_sum:>5d}/{n_sum} {ok_sum/n_sum:>6.1%}")
"""Experiment: sparse_metric="bm25" vs "tfidf" on the unchanged labeled sets.

D.2 from the production-hardening list: BM25 saturates term frequency and
normalizes by anchor length — often stronger when anchor lengths vary a lot.
Must NOT change anchors/test cases.
"""
import time

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

print("=" * 88)
print("TF-IDF vs BM25 sparse channel (nearest path, max+fixed defaults)")
print("=" * 88)
print(f"{'DATASET':8s} {'tfidf':>10s} {'bm25':>10s} {'bm25 boost=0.8':>14s} {'bm25 k1=1.2':>12s}")
print("-" * 88)
totals = {"tfidf": [0, 0], "bm25": [0, 0], "bm25_08": [0, 0], "bm25_k1_1": [0, 0]}
for name, options, tests in sets:
    tfidf = measure(options, tests, sparse_metric="tfidf")
    bm = measure(options, tests, sparse_metric="bm25")
    bm08 = measure(options, tests, sparse_metric="bm25", keyword_boost=0.8)
    bmk1 = measure(options, tests, sparse_metric="bm25", bm25_k1=1.0)
    for key, res in [("tfidf", tfidf), ("bm25", bm), ("bm25_08", bm08), ("bm25_k1_1", bmk1)]:
        totals[key][0] += res[0]
        totals[key][1] += res[1]
    print(f"{name:8s} {tfidf[0]:>7d}/{tfidf[1]:<2d} {bm[0]:>7d}/{bm[1]:<2d} "
          f"{bm08[0]:>11d}/{bm08[1]:<2d} {bmk1[0]:>10d}/{bmk1[1]:<2d}")
print("-" * 88)
for key, (ok, n) in totals.items():
    print(f"TOTAL {key:10s} {ok:>4d}/{n}  = {ok/n:.1%}")
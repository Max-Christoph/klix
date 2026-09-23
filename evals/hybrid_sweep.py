"""Experiment: classifier="hybrid" (learned dense+sparse fusion) vs. linear vs. nearest.

D.1 from the production-hardening list: instead of hand-tuning keyword_boost,
let a logistic probe learn the dense/sparse weighting on concatenated
[dense | tfidf] features. Must NOT change anchors/test cases.

Reuses the exact option sets and labeled cases from linear_sweep.py.
"""
import time

from linear_sweep import HR_OPTIONS, HR_TESTS, FIN_OPTIONS, FIN_TESTS, IMG_OPTIONS, IMG_CASES, TASK_OPTIONS, TASK_CASES, SHOP_OPTIONS, SHOP_CASES

from klix import DecisionEngine, Choice


def measure(options: dict, tests: list, classifier: str, **kw) -> tuple[int, int, float, float]:
    eng = DecisionEngine()
    eng.add_head(Choice(name="h", options=options, classifier=classifier, **kw))
    t0 = time.perf_counter()
    eng.compile()
    compile_ms = (time.perf_counter() - t0) * 1000
    ok = 0
    total_ms = 0.0
    for ticket, expected in tests:
        res = eng.decide(ticket)
        total_ms += res.latency_ms
        if res.h == expected:
            ok += 1
    return ok, len(tests), compile_ms, total_ms / max(1, len(tests))


sets = [
    ("HR", HR_OPTIONS, HR_TESTS),
    ("FIN", FIN_OPTIONS, FIN_TESTS),
    ("IMAGE", IMG_OPTIONS, [(q, e) for q, e in IMG_CASES if e]),
    ("TASK", TASK_OPTIONS, [(q, e) for q, e in TASK_CASES if e]),
    ("SHOP", SHOP_OPTIONS, [(q, e) for q, e in SHOP_CASES if e]),
]

print("=" * 92)
print("NEAREST vs LINEAR (dense-only probe) vs HYBRID (learned dense+sparse fusion)")
print("=" * 92)
print(f"{'DATASET':8s} {'nearest':>10s} {'linear':>10s} {'hybrid':>10s} {'hybrid C=1':>12s} {'hybrid ms':>10s}")
print("-" * 92)
totals = {"nearest": [0, 0], "linear": [0, 0], "hybrid": [0, 0]}
for name, options, tests in sets:
    near = measure(options, tests, "nearest")
    lin = measure(options, tests, "linear")
    hyb = measure(options, tests, "hybrid")
    hyb1 = measure(options, tests, "hybrid", classifier_C=1.0)
    for key, res in [("nearest", near), ("linear", lin), ("hybrid", hyb)]:
        totals[key][0] += res[0]
        totals[key][1] += res[1]
    print(f"{name:8s} {near[0]:>7d}/{near[1]:<2d} {lin[0]:>7d}/{lin[1]:<2d} "
          f"{hyb[0]:>7d}/{hyb[1]:<2d} {hyb1[0]:>9d}/{hyb1[1]:<2d} {hyb[3]:>10.1f}")
print("-" * 92)
for key, (ok, n) in totals.items():
    print(f"TOTAL {key:8s} {ok:>4d}/{n}  = {ok/n:.1%}")

# Latenz-Check: hybrid head eval cost vs linear
eng = DecisionEngine()
eng.add_head(Choice(name="a", options=HR_OPTIONS, classifier="hybrid"))
eng.add_head(Choice(name="b", options=FIN_OPTIONS, classifier="hybrid"))
eng.add_head(Choice(name="c", options=IMG_OPTIONS, classifier="hybrid"))
eng.compile()
enc = eng.backbone.encode("kurze testanfrage")
for _ in range(5):
    for h in eng.heads:
        h.evaluate(enc)
N = 500
t0 = time.perf_counter()
for _ in range(N):
    for h in eng.heads:
        h.evaluate(enc)
print(f"\nHybrid head eval (3 heads): {(time.perf_counter()-t0)/N*1000:.3f} ms")
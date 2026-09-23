"""Vergleich nearest-anchor vs. linear-probe classifier auf UNVERÄNDERTEN Daten.

Der lineare Klassifikator (classifier="linear") lernt eine Entscheidungsgrenze
auf den (augmentierten) Anker-Embeddings - das sollte verwandte Klassen trennen,
die reine Cosine-Similarity nicht auseinanderhält.
"""

import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from klix import DecisionEngine, Choice  # noqa: E402
from evals.eval_domains import (  # noqa: E402
    IMG_OPTIONS, TASK_OPTIONS, SHOP_OPTIONS, IMG_CASES, TASK_CASES, SHOP_CASES,
)
from evals.variant_sweep import HR_OPTIONS, FIN_OPTIONS, HR_TESTS, FIN_TESTS  # noqa: E402


def measure(options: dict, tests: list, classifier: str, **kw) -> tuple[int, int, float]:
    eng = DecisionEngine()
    eng.add_head(Choice(name="h", options=options, classifier=classifier, **kw))
    t0 = time.perf_counter()
    eng.compile()
    compile_ms = (time.perf_counter() - t0) * 1000
    ok = 0
    for ticket, expected in tests:
        got = eng.decide(ticket).h
        if got == expected:
            ok += 1
    return ok, len(tests), compile_ms


sets = [
    ("HR (12)", HR_OPTIONS, HR_TESTS),
    ("FIN (12)", FIN_OPTIONS, FIN_TESTS),
    ("IMAGE (12)", IMG_OPTIONS, [(q, e) for q, e in IMG_CASES if e]),
    ("TASK (12)", TASK_OPTIONS, [(q, e) for q, e in TASK_CASES if e]),
    ("SHOP (12)", SHOP_OPTIONS, [(q, e) for q, e in SHOP_CASES if e]),
]

print("=" * 84)
print("NEAREST-ANCHOR (Status quo) vs LINEAR-PROBE (classifier='linear')")
print("=" * 84)
print(f"{'DATASET':12s} {'nearest':>10s} {'linear':>10s} {'linear C=1':>12s} {'compile(ms)':>12s}")
print("-" * 84)
for name, options, tests in sets:
    near = measure(options, tests, "nearest")
    lin = measure(options, tests, "linear")
    lin_strong = measure(options, tests, "linear", classifier_C=1.0)
    print(f"{name:12s} {near[0]:>7d}/{near[1]:<2d} {lin[0]:>7d}/{lin[1]:<2d} "
          f"{lin_strong[0]:>9d}/{lin_strong[1]:<2d} {lin[2]:>11.1f}")

# Latenz: linear vs nearest bei Inference (nur Kopf, ohne Encoding)
print("\n--- Inference-Latenz (Kopf nur, 3 Köpfe) ---")
eng = DecisionEngine()
eng.add_head(Choice(name="a", options=HR_OPTIONS, classifier="linear"))
eng.add_head(Choice(name="b", options=FIN_OPTIONS, classifier="linear"))
eng.add_head(Choice(name="c", options=IMG_OPTIONS, classifier="linear"))
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
print(f"linear probe, 3 Köpfe: {(time.perf_counter() - t0) * 1000 / N:.4f} ms")

eng2 = DecisionEngine()
eng2.add_head(Choice(name="a", options=HR_OPTIONS))
eng2.add_head(Choice(name="b", options=FIN_OPTIONS))
eng2.add_head(Choice(name="c", options=IMG_OPTIONS))
eng2.compile()
enc2 = eng2.backbone.encode("kurze testanfrage")
for _ in range(5):
    for h in eng2.heads:
        h.evaluate(enc2)
t0 = time.perf_counter()
for _ in range(N):
    for h in eng2.heads:
        h.evaluate(enc2)
print(f"nearest (Status quo), 3 Köpfe: {(time.perf_counter() - t0) * 1000 / N:.4f} ms")
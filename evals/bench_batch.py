"""Honest batch benchmark: decide_batch vs serial decide at multiple sizes.

Answers the question: does the batch path actually amortize the model-call
overhead, or does it secretly loop? Expectation if batching works:
- small batches: little gain (ONNX forward cost dominates)
- large batches: per-item cost drops toward the pure compute floor
"""

import sys
import time

sys.path.insert(0, "src")

from klix import Choice, DecisionEngine, Score


def build():
    eng = DecisionEngine()
    eng.add_head(Choice(
        name="route",
        options={
            "billing": ["approve invoice", "cost center over budget"],
            "technical": ["server down", "vpn keeps dropping"],
        },
    ))
    eng.add_head(Score(
        name="urgency",
        low_anchors=["routine maintenance", "casual question"],
        high_anchors=["emergency right now", "production line down"],
    ))
    eng.compile()
    return eng


def main():
    eng = build()
    eng.decide("warmup ticket about the server")

    print(f"{'n':>6} | {'serial ms':>10} {'ms/item':>8} | {'batch ms':>9} {'ms/item':>8} | {'speedup':>7}")
    print("-" * 68)
    for n in [1, 5, 25, 100, 500]:
        texts = [f"ticket number {i} about the server and vpn problem {i}" for i in range(n)]

        # Serial
        t0 = time.perf_counter()
        for t in texts:
            eng.decide(t)
        serial_ms = (time.perf_counter() - t0) * 1000

        # Batch (median of 3 runs for stability)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            eng.decide_batch(texts)
            times.append((time.perf_counter() - t0) * 1000)
        batch_ms = sorted(times)[1]

        speedup = serial_ms / batch_ms if batch_ms > 0 else float("inf")
        print(f"{n:>6} | {serial_ms:>9.1f}  {serial_ms/n:>7.2f} | {batch_ms:>8.1f}  {batch_ms/n:>7.2f} | {speedup:>6.2f}x")


if __name__ == "__main__":
    main()
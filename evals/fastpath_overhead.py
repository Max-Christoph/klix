"""Measure the fast-path miss penalty WITHOUT relying on the noisy dense pass.

The v0.8.8 fast-path miss paid for a SECOND query vectorization. Wall-clock
comparison of miss vs baseline cannot resolve that here, because the ONNX dense
forward pass (~10-30 ms, and its sign flips between runs on a loaded host)
drowns the effect being measured. So measure the mechanism directly:

  cost of one `_sparse_state` build  x  (2 builds in v0.8.8 - 1 build now)

Run: uv run python -m evals.fastpath_overhead
"""
import statistics
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from klix import Choice, DecisionEngine, manufacturing_glossary  # noqa: E402

OPT = {
    "downtime": ["conveyor belt stopped", "line is down", "cycle time doubled",
                 "machine stopped without error"],
    "maintenance": ["spare part missing", "calibration overdue",
                    "schedule maintenance for the line", "sensor replacement"],
}
QUERIES = [
    "das foerderband steht seit heute morgen",
    "die taktzeit hat sich nach dem neustart verdoppelt",
    "ersatzteil fuer die hydraulikeinheit fehlt",
    "totally unrelated content about holidays and hobbies",
    "i would like to book a table for dinner tonight",
]


def main() -> None:
    eng = DecisionEngine(glossary=manufacturing_glossary(), sparse_fastpath=True)
    eng.add_head(Choice(name="route", options=OPT))
    eng.compile()
    head = eng.heads[0]

    # Warm up, then time ONLY the sparse-state build (the thing v0.8.8 did twice).
    for t in QUERIES:
        head._sparse_state(t)
    samples = []
    for _ in range(60):
        for t in QUERIES:
            t0 = time.perf_counter()
            head._sparse_state(t)
            samples.append((time.perf_counter() - t0) * 1000)

    samples.sort()
    med = statistics.median(samples)
    p95 = samples[int(0.95 * (len(samples) - 1))]
    print("SPARSE-STATE BUILD (the eliminated duplicate work)")
    print(f"  median {med:.4f} ms   p95 {p95:.4f} ms   n={len(samples)}")
    print(f"  min    {samples[0]:.4f} ms   max {samples[-1]:.4f} ms")
    print()
    print("  v0.8.8 on a fast-path MISS : 2 builds  (gate + evaluate)")
    print("  v0.9.0 on a fast-path MISS : 1 build   (shared SparseQuery)")
    print(f"  -> removed overhead per miss: ~{med:.4f} ms median "
          f"(the query is vectorized once either way now)")
    print()
    print("  Why not wall-clock miss-vs-baseline: the ONNX dense forward pass")
    print("  dominates and its own variance exceeds this effect, so the measured")
    print("  SIGN of the difference flips between runs on a loaded host. The")
    print("  build COUNT is the deterministic evidence; this is its cost.")


if __name__ == "__main__":
    main()

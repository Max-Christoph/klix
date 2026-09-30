"""Isolated measurements for the README benchmark table.

Run this ALONE — no test suite, no other sweep. Timing measured under CPU
contention is worthless, and that mistake has already been made once in this
repo (a `< 20 ms` assertion failed at 40.1 ms purely because a model sweep was
running alongside).

Measures, each from a clean process where it matters:
  1. single-decision latency (p50/p95) — warm, after a warm-up call
  2. resident memory of a process with the engine compiled and the model loaded
  3. the on-disk model footprint (the number "118 MB" in evals/backbone_shootout.py
     turned out to be a label, not a measurement: the ONNX file alone is 235 MB)

Run:
    python evals/measure_footprint.py
"""
from __future__ import annotations

import os
import statistics
import sys
import time

sys.path.insert(0, "src")

from klix import Choice, DecisionEngine  # noqa: E402

TEXTS = [
    "plc-34 reports a fault, conveyor belt stopped",
    "vpn keeps dropping since the update",
    "invoice 4711 needs approval",
    "robot cell stopped mid cycle",
    "oil spill in hall 2",
]


def build() -> DecisionEngine:
    eng = DecisionEngine()
    eng.add_head(Choice(name="queue", options={
        "it_ops": ["vpn down", "server unreachable", "laptop won't boot"],
        "ot_plant": ["robot cell stopped", "PLC fault", "cycle time deviation"],
        "facility": ["oil spill in hall 2", "heating broken"],
    }))
    eng.compile()
    return eng


def rss_mb() -> float:
    """Resident set size in MB. Windows has no resource.getrusage(RUSAGE_SELF)."""
    try:
        import ctypes
        import ctypes.wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", ctypes.wintypes.DWORD),
                        ("PageFaultCount", ctypes.wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters),
            counters.cb)
        return counters.WorkingSetSize / 1024 / 1024
    except Exception:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            return float("nan")


def model_files_mb() -> tuple[float, list[tuple[str, float]]]:
    from fastembed import TextEmbedding
    eng_model = TextEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    total, files = 0.0, []
    d = getattr(eng_model.model, "_model_dir", None)
    if d is None:
        return float("nan"), []
    for f in d.iterdir():
        if f.is_file():
            mb = f.stat().st_size / 1024 / 1024
            total += mb
            files.append((f.name, round(mb, 1)))
    return total, files


def main() -> None:
    before = rss_mb()

    eng = build()
    eng.decide(TEXTS[0])          # warm-up: first call loads the ONNX session

    after_load = rss_mb()

    lat = []
    for i in range(60):
        t0 = time.perf_counter()
        eng.decide(TEXTS[i % len(TEXTS)])
        lat.append((time.perf_counter() - t0) * 1000)

    lat.sort()
    p50 = statistics.median(lat)
    p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]

    print("=" * 62)
    print("klix — isolated measurements (clean process)")
    print("=" * 62)
    print(f"  decide() p50           : {p50:6.1f} ms")
    print(f"  decide() p95           : {p95:6.1f} ms")
    print(f"  decide() min / max     : {lat[0]:.1f} / {lat[-1]:.1f} ms")
    print(f"  RSS after compile+load : {after_load:6.1f} MB")
    print(f"  RSS before             : {before:6.1f} MB")
    print(f"  RSS delta (engine)     : {after_load - before:6.1f} MB")

    total, files = model_files_mb()
    print(f"\n  embedding model on disk: {total:.1f} MB")
    for name, mb in files:
        print(f"      {name:28} {mb:7.1f} MB")

    print("\n  NOTE for the README: '118 MB' in evals/backbone_shootout.py is a")
    print("  hardcoded label, not a measurement. The figure above is measured.")


if __name__ == "__main__":
    main()

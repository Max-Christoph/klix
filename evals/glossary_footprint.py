"""Measure glossary size/load curves (v0.9.0 footprint tuning).

Determines the concept count at which the JSON parse + construction stays under
the 20 ms load budget. Run: uv run python -m evals.glossary_footprint
"""
import json
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from klix.glossary import DEFAULT_GLOSSARY, Glossary  # noqa: E402


def deep_size(o, seen=None):
    if seen is None:
        seen = set()
    if id(o) in seen:
        return 0
    seen.add(id(o))
    s = sys.getsizeof(o)
    if isinstance(o, dict):
        s += sum(deep_size(k, seen) + deep_size(v, seen) for k, v in o.items())
    elif isinstance(o, (list, tuple, set, frozenset)):
        s += sum(deep_size(v, seen) for v in o)
    return s


def main() -> None:
    full = json.loads(DEFAULT_GLOSSARY.read_text(encoding="utf-8"))
    print(f"bundled file: {len(full)} concepts")
    print(f"{'concepts':>9} {'terms':>7} {'KB':>8} {'parse':>9} {'ctor':>8} {'index':>9} {'deep MB':>8}")
    for n in (5000, 6000, 7000, 8000, 10000, 12000, len(full)):
        sub = {k: full[k] for k in list(full)[:n]}
        blob = json.dumps(sub, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        nb = len(blob.encode("utf-8"))
        terms = sum(len(t) for v in sub.values() for t in v.values())
        t0 = time.perf_counter()
        d = json.loads(blob)
        tp = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        g = Glossary(d, index=False)
        tc = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        g._ensure_index()
        ti = (time.perf_counter() - t0) * 1000
        total = tp + tc
        verdict = "OK" if total < 20 else "over"
        print(f"{n:9d} {terms:7d} {nb/1024:8.0f} {tp:8.1f}ms {tc:7.2f}ms {ti:8.1f}ms "
              f"{deep_size(d)/1048576:7.2f}  load={total:5.1f}ms {verdict}")


if __name__ == "__main__":
    main()

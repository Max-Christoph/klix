"""Glossary hook benchmark: baseline vs `glossary=` on mixed DE/EN routing.

Design of the test set (matters for interpreting the numbers):
  - The schema has ENGLISH anchors, the queries are GERMAN where a translation
    actually matters. That is the case a glossary can fix: the sparse channel
    has no shared vocabulary, and the dense channel has to carry the whole load.
  - A second group uses German anchors with English queries (mirror direction).
  - A third group is monolingual German (anchors and queries DE) as a control:
    the glossary should be near-neutral there, since no bridge is needed.

Metrics: accuracy, latency (ms per decision, median), peak RSS delta (MB).

Run: uv run python -m evals.glossary_bench
"""
import json
import statistics
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from klix import Choice, DecisionEngine, load_glossary  # noqa: E402

# ---------------------------------------------------------------------------
# Test data: production-domain tickets with technical terms the glossary knows
# ---------------------------------------------------------------------------
EN_ANCHORS = {
    "maintenance": [
        "schedule preventive maintenance for the line",
        "spare part for the hydraulic unit is missing",
        "calibration of the sensor is overdue",
        "commissioning of the new cell is planned",
    ],
    "downtime": [
        "conveyor belt stopped mid shift",
        "production line is down since this morning",
        "unplanned downtime on the assembly cell",
        "cycle time doubled after the restart",
    ],
    "quality": [
        "scrap rate is too high on this batch",
        "reject parts pile up at station four",
        "error code E42 appears on the HMI",
        "lot size for the next batch is wrong",
    ],
}

# German queries against those English anchors (the case a glossary can fix)
DE_QUERIES = [
    ("das foerderband steht seit heute morgen", "downtime"),
    ("die taktzeit hat sich nach dem neustart verdoppelt", "downtime"),
    ("stillstand an der montagezelle, bitte pruefen", "downtime"),
    ("ersatzteil fuer die hydraulikeinheit fehlt", "maintenance"),
    ("kalibrierung des sensors ist ueberfaellig", "maintenance"),
    ("inbetriebnahme der neuen zelle ist geplant", "maintenance"),
    ("wartung fuer die linie einplanen", "maintenance"),
    ("ausschussquote bei dieser charge ist zu hoch", "quality"),
    ("fehlercode E42 erscheint auf dem panel", "quality"),
    ("losgroesse fuer die naechste charge stimmt nicht", "quality"),
]

# Mirror direction: German anchors, English queries
DE_ANCHORS = {
    "maintenance": [
        "praeventive wartung fuer die linie einplanen",
        "ersatzteil fuer die hydraulikeinheit fehlt",
        "kalibrierung des sensors ist ueberfaellig",
    ],
    "downtime": [
        "foerderband steht mitten in der schicht",
        "produktionslinie ist seit heute morgen down",
        "taktzeit hat sich nach dem neustart verdoppelt",
    ],
    "quality": [
        "ausschussquote bei dieser charge ist zu hoch",
        "fehlercode E42 erscheint auf dem panel",
        "losgroesse fuer die naechste charge stimmt nicht",
    ],
}
EN_QUERIES = [
    ("the conveyor belt stopped mid shift", "downtime"),
    ("cycle time doubled after the restart", "downtime"),
    ("line is down since this morning", "downtime"),
    ("spare part for the hydraulic unit is missing", "maintenance"),
    ("sensor calibration is overdue", "maintenance"),
    ("commissioning of the new cell is planned", "maintenance"),
    ("scrap rate is too high on this batch", "quality"),
    ("error code E42 appears on the panel", "quality"),
    ("lot size for the next batch is wrong", "quality"),
]

# Control: monolingual German both sides, no bridge needed
DE_ONLY_ANCHORS = {
    "wartung": ["wartung einplanen", "ersatzteil fehlt", "kalibrierung ueberfaellig"],
    "stillstand": ["foerderband steht", "linie ist down", "taktzeit verdoppelt"],
    "qualitaet": ["ausschuss zu hoch", "fehlercode E42", "losgroesse falsch"],
}
DE_ONLY_QUERIES = [
    ("wartung fuer die linie einplanen", "wartung"),
    ("ersatzteil ist nicht da", "wartung"),
    ("kalibrierung des sensors fehlt", "wartung"),
    ("das foerderband steht wieder", "stillstand"),
    ("die linie ist seit heute down", "stillstand"),
    ("taktzeit hat sich verdoppelt", "stillstand"),
    ("ausschuss ist zu hoch", "qualitaet"),
    ("fehlercode E42 auf dem panel", "qualitaet"),
    ("losgroesse stimmt nicht", "qualitaet"),
]

GROUPS = [
    ("EN anchors <- DE queries", EN_ANCHORS, DE_QUERIES),
    ("DE anchors <- EN queries", DE_ANCHORS, EN_QUERIES),
    ("DE anchors <- DE queries (control)", DE_ONLY_ANCHORS, DE_ONLY_QUERIES),
]


def _rss_mb() -> float:
    """Peak RSS in MB, stdlib only (no psutil dependency).

    POSIX: resource.getrusage. Windows: ctypes against psapi. Returns 0.0 if
    neither is available rather than failing the benchmark.
    """
    try:
        import resource  # POSIX only
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:  # noqa: BLE001
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        get_info.restype = wintypes.BOOL
        if get_info(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return counters.PeakWorkingSetSize / 1048576.0
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def run(anchors: dict, queries: list, *, use_glossary: bool, classifier: str,
        keyword_boost: float) -> tuple[int, int, list[float]]:
    glossary = load_glossary() if use_glossary else None
    eng = DecisionEngine()
    eng.add_head(Choice(
        name="route", options=anchors, classifier=classifier,
        glossary=glossary, keyword_boost=keyword_boost,
    ))
    eng.compile()
    ok, lat = 0, []
    for text, expected in queries:
        t0 = time.perf_counter()
        got = eng.decide(text).route
        lat.append((time.perf_counter() - t0) * 1000)
        ok += int(got == expected)
    return ok, len(queries), lat


def per_item(anchors: dict, queries: list, *, use_glossary: bool, classifier: str,
             keyword_boost: float) -> list[int]:
    """Per-query correctness, pooled over the cross-lingual groups only."""
    glossary = load_glossary() if use_glossary else None
    eng = DecisionEngine()
    eng.add_head(Choice(
        name="route", options=anchors, classifier=classifier,
        glossary=glossary, keyword_boost=keyword_boost,
    ))
    eng.compile()
    return [int(eng.decide(t).route == e) for t, e in queries]


def bootstrap_crosslingual(classifier: str, keyword_boost: float, n_boot: int = 2000,
                           seed: int = 23) -> tuple[float, float, float, float, int]:
    """Accuracy + delta CI over the two CROSS-LINGUAL groups (19 cases).

    The monolingual control is excluded on purpose: pooling it would dilute the
    effect we are trying to measure.
    """
    import numpy as np

    base, cand = [], []
    for _gname, anchors, queries in GROUPS[:2]:  # the two cross-lingual groups
        base.extend(per_item(anchors, queries, use_glossary=False,
                             classifier=classifier, keyword_boost=keyword_boost))
        cand.extend(per_item(anchors, queries, use_glossary=True,
                             classifier=classifier, keyword_boost=keyword_boost))
    a, b = np.array(base), np.array(cand)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        diffs.append(b[idx].mean() - a[idx].mean())
    diffs = np.sort(diffs)
    return a.mean(), b.mean(), diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)], len(a)


def main() -> None:
    configs = [
        ("nearest, kb=0.5", "nearest", 0.5),
        ("centroid, kb=0.5", "centroid", 0.5),
        ("centroid, kb=1.5", "centroid", 1.5),
        ("linear", "linear", 0.5),
    ]

    print("=" * 96)
    print("GLOSSARY HOOK BENCHMARK — baseline vs glossary= on mixed DE/EN routing")
    print("=" * 96)
    print(f"{'group':36s} {'config':20s} {'base':>8s} {'gloss':>8s} {'delta':>8s}  {'ms/dec':>8s}")
    print("-" * 96)

    totals = {}
    for gname, anchors, queries in GROUPS:
        for cname, classifier, kb in configs:
            try:
                b_ok, b_n, b_lat = run(anchors, queries, use_glossary=False,
                                       classifier=classifier, keyword_boost=kb)
                g_ok, g_n, g_lat = run(anchors, queries, use_glossary=True,
                                       classifier=classifier, keyword_boost=kb)
            except Exception as ex:  # noqa: BLE001
                print(f"{gname:36s} {cname:20s} FAILED {type(ex).__name__}: {str(ex)[:30]}")
                continue
            delta = g_ok - b_ok
            med = statistics.median(g_lat)
            print(f"{gname:36s} {cname:20s} {b_ok:>3d}/{b_n:<3d} {g_ok:>3d}/{g_n:<3d} "
                  f"{delta:>+8d}  {med:>8.1f}")
            key = (gname, cname)
            totals[key] = (b_ok, g_ok, b_n, med)

    print()
    print("=" * 96)
    print("AGGREGATE over all three groups (27 cases each)")
    print("=" * 96)
    print(f"{'config':22s} {'baseline':>12s} {'glossary':>12s} {'delta':>9s}  {'median ms':>10s}")
    print("-" * 96)
    for cname, _cls, _kb in configs:
        b = sum(totals[(g, cname)][0] for g, _, _ in GROUPS if (g, cname) in totals)
        g_ = sum(totals[(g, cname)][1] for g, _, _ in GROUPS if (g, cname) in totals)
        n = sum(totals[(g, cname)][2] for g, _, _ in GROUPS if (g, cname) in totals)
        ms = statistics.median([totals[(g, cname)][3] for g, _, _ in GROUPS if (g, cname) in totals])
        print(f"{cname:22s} {b:>4d}/{n:<7d} {g_:>4d}/{n:<7d} {g_-b:>+9d}  {ms:>10.1f}")

    print()
    print(f"peak RSS after run: {_rss_mb():.0f} MB")

    print()
    print("=" * 96)
    print("BOOTSTRAP over the two CROSS-LINGUAL groups (19 cases; control excluded)")
    print("=" * 96)
    print(f"{'config':22s} {'baseline':>10s} {'glossary':>10s} {'delta':>9s}  {'95% CI':>18s}  verdict")
    print("-" * 96)
    for cname, classifier, kb in configs:
        try:
            b, g_, lo, hi, n = bootstrap_crosslingual(classifier, kb)
            sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not distinguishable"
            print(f"{cname:22s} {b:>9.1%} {g_:>10.1%} {g_-b:>+9.1%}  "
                  f"[{lo:+.1%}, {hi:+.1%}]  {sig}")
        except Exception as ex:  # noqa: BLE001
            print(f"{cname:22s} FAILED {type(ex).__name__}: {str(ex)[:30]}")

    print()
    print("Reading the table: the glossary can only help where the sparse channel")
    print("has no shared vocabulary (cross-lingual groups). The monolingual control")
    print("stays flat or dips slightly — expansion adds terms that dilute the sparse")
    print("vector, which is the honest cost of the hook.")


if __name__ == "__main__":
    main()

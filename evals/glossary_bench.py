"""Glossary hook benchmark: baseline vs weighted glossary vs fast path.

Extends the v0.8.7 benchmark with the v0.8.8 features:
  - alpha sweep for `glossary_weight` (weighted expansion)
  - fast path (sparse early exit) with latency comparison and hit rate
  - explicit regression check on the MONOLINGUAL control group

Test set design (matters for reading the numbers):
  - EN anchors <- DE queries: the case a glossary can fix (sparse channel has
    no shared vocabulary, so the dense channel had to carry everything).
  - DE anchors <- EN queries: the mirror direction.
  - DE anchors <- DE queries: monolingual control. No bridge needed, so the
    v0.8.7 concatenation DILUTED the query and cost accuracy. This is the case
    weighted expansion is supposed to fix — verified explicitly below.

Metrics: accuracy, median latency per decision (ms), peak RSS (MB).

Run: uv run python -m evals.glossary_bench
"""
import statistics
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from klix import Choice, DecisionEngine, Glossary, load_glossary  # noqa: E402

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
    """Peak RSS in MB, stdlib only (no psutil)."""
    try:
        import resource  # POSIX
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:  # noqa: BLE001
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        c = PMC()
        c.cb = ctypes.sizeof(c)
        fn = ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        fn.restype = wintypes.BOOL
        if fn(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return c.PeakWorkingSetSize / 1048576.0
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def build(anchors: dict, *, use_glossary: bool, classifier: str, keyword_boost: float,
          glossary_weight: float, fastpath) -> DecisionEngine:
    eng = DecisionEngine(sparse_fastpath=fastpath)
    eng.add_head(Choice(
        name="route", options=anchors, classifier=classifier,
        keyword_boost=keyword_boost,
        glossary=load_glossary() if use_glossary else None,
        glossary_weight=glossary_weight,
    ))
    eng.compile()
    return eng


def run(anchors: dict, queries: list, **kw):
    eng = build(anchors, **kw)
    ok, lat = 0, []
    for text, expected in queries:
        t0 = time.perf_counter()
        got = eng.decide(text).route
        lat.append((time.perf_counter() - t0) * 1000)
        ok += int(got == expected)
    return ok, len(queries), lat, eng.fastpath_stats()


def show(group, queries, label, **kw):
    anchors = group
    try:
        ok, n, lat, stats = run(anchors, queries, **kw)
        med = statistics.median(lat)
        extra = f"  fp {stats['hits']}/{stats['hits']+stats['misses']}" if stats["enabled"] else ""
        print(f"  {label:34s} {ok:3d}/{n} = {ok/n:6.1%}  {med:6.1f} ms{extra}")
        return ok, n, med, stats
    except Exception as ex:  # noqa: BLE001
        print(f"  {label:34s} FAILED {type(ex).__name__}: {str(ex)[:38]}")
        return None


def main() -> None:
    print("=" * 100)
    print("GLOSSARY / FAST-PATH BENCHMARK (v0.8.8)")
    print("=" * 100)

    cfg = dict(classifier="nearest", keyword_boost=0.5, glossary_weight=0.4, fastpath=None)

    print("\n########## 1. ALPHA SWEEP (weighted expansion) ##########")
    print("   alpha=0.0 disables the glossary on the QUERY side (anchors stay expanded)")
    for gname, anchors, queries in GROUPS:
        print(f"\n-- {gname} --")
        show(anchors, queries, "no glossary (baseline)", **{**cfg, "use_glossary": False})
        for a in (0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
            show(anchors, queries, f"glossary alpha={a}", **{**cfg, "use_glossary": True,
                                                              "glossary_weight": a})

    print("\n########## 2. FAST PATH (sparse early exit) ##########")
    print("   fp = fast-path hits / total decisions")
    for gname, anchors, queries in GROUPS:
        print(f"\n-- {gname} --")
        show(anchors, queries, "dense only (no fastpath)", **{**cfg, "use_glossary": True})
        ok, n, med, stats = show(anchors, queries, "fastpath enabled",
                                 **{**cfg, "use_glossary": True, "fastpath": True}) or (None,)*4
    print()
    print("   Latency for a fast-path HIT vs a dense MISS (same schema):")
    eng = build(EN_ANCHORS, use_glossary=True, classifier="nearest",
                keyword_boost=0.5, glossary_weight=0.4, fastpath=True)
    hit_lat, miss_lat = [], []
    for text, _ in DE_QUERIES:
        t0 = time.perf_counter()
        res = eng.decide(text)
        dt = (time.perf_counter() - t0) * 1000
        (hit_lat if res.details("route")["engine"] == "sparse_fastpath" else miss_lat).append(dt)
    if hit_lat:
        print(f"     fast-path hit : {statistics.median(hit_lat):6.2f} ms median (n={len(hit_lat)})")
    if miss_lat:
        print(f"     dense fallback: {statistics.median(miss_lat):6.2f} ms median (n={len(miss_lat)})")
    if hit_lat and miss_lat:
        sp = statistics.median(miss_lat) / statistics.median(hit_lat)
        print(f"     speedup on hits: {sp:.1f}x")

    print("\n########## 3. MONOLINGUAL REGRESSION CHECK ##########")
    print("   v0.8.7 concatenation cost accuracy here; weighted expansion must not")
    ctrl = GROUPS[2]
    base = show(ctrl[1], ctrl[2], "baseline (no glossary)", **{**cfg, "use_glossary": False})
    for a in (0.0, 0.3, 0.4, 0.5, 1.0):
        got = show(ctrl[1], ctrl[2], f"glossary alpha={a}", **{**cfg, "use_glossary": True,
                                                                 "glossary_weight": a})
        if base and got:
            verdict = "OK (no regression)" if got[0] >= base[0] else "REGRESSION"
            print(f"      -> vs baseline: {got[0]-base[0]:+d} case(s)  {verdict}")

    print("\n########## 4. GLOSSARY API ##########")
    custom = Glossary.load({"press": {"de": ["presse"], "en": ["press"]}})
    merged = load_glossary().merge(custom)
    print(f"   built-in entries : {len(load_glossary().mapping)}")
    print(f"   custom entries   : {len(custom.mapping)}")
    print(f"   merged entries   : {len(merged.mapping)}  ('press' present: {'press' in merged.mapping})")

    print(f"\npeak RSS: {_rss_mb():.0f} MB")


if __name__ == "__main__":
    main()

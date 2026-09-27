"""Diagnose ALL round-trip failures: mapping error or thin-anchor artifact?

The round-trip test (`evals/curated_glossary_verify.py`) uses ONE German word per
concept against anchors built from that concept's English terms only (1-3 words,
no sentence). Real schemas have sentence anchors. So a failure can be either:

  (a) MAPPING ERROR  — the German term is not linked to the concept at all
                       (the glossary never bridges it)
  (b) THIN ANCHOR    — the term IS linked and the keyword channel fires, but the
                       dense channel (which sees a single rare word against 1-3
                       word anchors) outvotes it
  (c) SPARSE ALSO WRONG — bridged and the keyword channel itself picks another
                       concept, which IS a genuine defect

This script discovers the failures instead of listing them. An earlier revision
carried a hard-coded list of 10 cases captured when the curated set had 124
concepts; after the expansion to 362 the round-trip count grew to 73 while the
diagnosis kept reporting the same 10, so 63 failures were never classified. The
failure set is now derived from the shipped artifact, which is also what makes
the verdict summary (one line per class) a complete statement.

Run: uv run python -m evals.roundtrip_failure_diag
"""
import importlib.util
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from klix import Choice, DecisionEngine  # noqa: E402
from klix.glossary import Glossary  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
# The content module imports its sibling `curated_glossary_ext`, so scripts/ must
# be importable — loading by spec path alone is not enough.
sys.path.insert(0, str(REPO / "scripts"))
_spec = importlib.util.spec_from_file_location(
    "curated", REPO / "scripts" / "curated_glossary_data.py")
curated = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(curated)

VERDICTS = (
    "MAPPING ERROR (glossary does not bridge the probe)",
    "SPARSE ALSO WRONG (bridged, keyword channel picks another concept)",
    "THIN ANCHOR (sparse picks right, dense outvotes)",
)


def diagnose(domain: str, mapping: dict, concept: str, probe: str, verbose: bool = True) -> str:
    g = Glossary(dict(mapping))
    anchors = {c: v["en"] for c, v in mapping.items()}
    eng = DecisionEngine(glossary=g)
    eng.add_head(Choice(name="r", options=anchors))
    eng.compile()
    head = eng.heads[0]

    got = eng.decide(probe).r
    # 1. does the glossary bridge this probe at all?
    concepts_hit = g.match_concepts(probe)
    expanded = head.expand_query_terms(probe)
    # 2. what does the sparse channel alone say?
    sparse_state = head._sparse_state(probe)
    row = sparse_state.csr()
    if row is not None:
        sims = np.asarray((row @ head.sparse_matrix.T).todense()).ravel()
    else:
        sims = np.zeros(head.sparse_matrix.shape[0])
    by_label: dict[str, float] = {}
    for i, lab in enumerate(head.label_map):
        by_label[lab] = max(by_label.get(lab, 0.0), float(sims[i]))
    sparse_rank = sorted(by_label.items(), key=lambda kv: -kv[1])[:3]
    # 3. dense ranking
    enc = eng.backbone.encode(probe)
    dsims = head.dense_matrix @ enc.dense_vec
    dby: dict[str, float] = {}
    for i, lab in enumerate(head.label_map):
        dby[lab] = max(dby.get(lab, 0.0), float(dsims[i]))
    dense_rank = sorted(dby.items(), key=lambda kv: -kv[1])[:3]

    if not concepts_hit:
        verdict = VERDICTS[0]
    elif sparse_rank[0][0] != concept:
        verdict = VERDICTS[1]
    else:
        verdict = VERDICTS[2]

    if verbose:
        print(f"\n[{domain}] {concept}  probe={probe!r}  -> decided {got!r}")
        print(f"  glossary concepts hit : {concepts_hit}")
        print(f"  expansion of probe    : {expanded}")
        print(f"  sparse top3           : {[(k, round(v, 3)) for k, v in sparse_rank]}")
        print(f"  dense  top3           : {[(k, round(v, 3)) for k, v in dense_rank]}")
        print(f"  VERDICT               : {verdict}")
    return verdict


def collect_failures() -> list[tuple[str, str, str]]:
    """Every (domain, concept, probe) whose German probe misses its own concept."""
    out: list[tuple[str, str, str]] = []
    for domain, mapping in curated.all_domains().items():
        g = Glossary(dict(mapping))
        anchors = {c: v["en"] for c, v in mapping.items()}
        eng = DecisionEngine(glossary=g)
        eng.add_head(Choice(name="r", options=anchors))
        eng.compile()
        for concept, langs in mapping.items():
            probe = langs["de"][0]
            if eng.decide(probe).r != concept:
                out.append((domain, concept, probe))
    return out


def main() -> None:
    domains = curated.all_domains()
    failures = collect_failures()

    print("=" * 100)
    print("ROUND-TRIP FAILURE DIAGNOSIS")
    print("=" * 100)
    print(f"\nfailures discovered from the shipped artifact: {len(failures)}")

    counts: Counter[str] = Counter()
    details: dict[str, list[tuple[str, str, str]]] = {v: [] for v in VERDICTS}
    for domain, concept, probe in failures:
        verdict = diagnose(domain, domains[domain], concept, probe, verbose=False)
        counts[verdict] += 1
        details[verdict].append((domain, concept, probe))

    print("\n" + "=" * 100)
    print("VERDICT SUMMARY (all failures classified)")
    print("=" * 100)
    for verdict in VERDICTS:
        print(f"  {counts[verdict]:4d}  {verdict}")

    genuine = counts[VERDICTS[0]] + counts[VERDICTS[1]]
    print(f"\n  genuine mapping defects (a + b): {genuine}")
    print(f"  thin-anchor artifacts        (c): {counts[VERDICTS[2]]}")

    for verdict in (VERDICTS[0], VERDICTS[1]):
        if details[verdict]:
            print(f"\n  {verdict}:")
            for domain, concept, probe in details[verdict]:
                print(f"    [{domain}] {concept:22s} probe={probe!r}")

    print("\n  first 3 thin-anchor cases in full (mechanism):")
    for domain, concept, probe in details[VERDICTS[2]][:3]:
        diagnose(domain, domains[domain], concept, probe, verbose=True)


if __name__ == "__main__":
    main()

"""Diagnose the 10 round-trip failures: mapping error or thin-anchor artifact?

The round-trip test uses ONE German word per concept against anchors built from
that concept's English terms only (1-3 words, no sentence). Real schemas have
sentence anchors. So a failure can be either:

  (a) MAPPING ERROR  — the German term is not linked to the concept at all
  (b) THIN ANCHOR    — the term IS linked and the keyword channel fires, but the
                       dense channel (which sees a single rare word against 1-3
                       word anchors) outvotes it

This prints, per failure: whether the glossary expands the probe, the sparse and
dense contributions, and which concept the sparse channel alone would pick. If
the sparse channel picks the RIGHT concept, it is (b).

Run: uv run python -m evals.roundtrip_failure_diag
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402

from klix import Choice, DecisionEngine  # noqa: E402
from klix.glossary import Glossary  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "curated", REPO / "scripts" / "curated_glossary_data.py")
curated = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curated)

FAILURES = [
    ("manufacturing", "cycle_time", "taktzeit"),
    ("manufacturing", "voltage", "spannung"),
    ("manufacturing", "switch", "schalter"),
    ("manufacturing", "inspection", "pruefung"),
    ("manufacturing", "failure_cause", "stoergrund"),
    ("it", "backup", "sicherungskopie"),
    ("everyday", "salary", "gehalt"),
    ("everyday", "receipt", "quittung"),
    ("everyday", "supplier", "lieferant"),
    ("everyday", "certificate_hr", "zeugnis"),
]


def main():
    domains = curated.all_domains()
    print("=" * 100)
    print("ROUND-TRIP FAILURE DIAGNOSIS")
    print("=" * 100)

    for domain, concept, probe in FAILURES:
        mapping = domains[domain]
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
        import scipy.sparse as sp
        row = sparse_state.csr()
        if row is not None:
            sims = np.asarray((row @ head.sparse_matrix.T).todense()).ravel()
        else:
            sims = np.zeros(head.sparse_matrix.shape[0])
        by_label = {}
        for i, lab in enumerate(head.label_map):
            by_label[lab] = max(by_label.get(lab, 0.0), float(sims[i]))
        sparse_rank = sorted(by_label.items(), key=lambda kv: -kv[1])[:3]
        # 3. dense ranking
        enc = eng.backbone.encode(probe)
        dsims = head.dense_matrix @ enc.dense_vec
        dby = {}
        for i, lab in enumerate(head.label_map):
            dby[lab] = max(dby.get(lab, 0.0), float(dsims[i]))
        dense_rank = sorted(dby.items(), key=lambda kv: -kv[1])[:3]

        verdict = "MAPPING ERROR" if not concepts_hit else (
            "THIN ANCHOR (sparse picks right, dense outvotes)"
            if sparse_rank[0][0] == concept else
            "SPARSE ALSO WRONG")
        print(f"\n[{domain}] {concept}  probe={probe!r}  -> decided {got!r}")
        print(f"  glossary concepts hit : {concepts_hit}")
        print(f"  expansion of probe    : {expanded}")
        print(f"  sparse top3           : {[(k, round(v,3)) for k,v in sparse_rank]}")
        print(f"  dense  top3           : {[(k, round(v,3)) for k,v in dense_rank]}")
        print(f"  VERDICT               : {verdict}")


if __name__ == "__main__":
    main()

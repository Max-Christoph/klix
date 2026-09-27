"""Verification of the curated tri-domain glossary — per domain, all entries.

The task asked for a random sample of 100-150 terms across the three domains.
The curated list has 124 concepts, so this checks ALL of them: no sampling error.

Three independent checks per concept:

  1. STRUCTURAL — the engine's own rules (duplicate terms, ambiguous mappings,
     homographs, anchor collisions) via Glossary.validate(). Mechanical, not
     opinion: a failure here is an unambiguous defect.

  2. CROSS-LINGUAL AGREEMENT — embed the German and the English side with the
     backbone klix actually routes with and measure cosine. A pair naming the
     same concept should agree.

     LIMITATION, stated up front: this metric penalises RARE terms, not wrong
     ones. It previously flagged 'lunar eclipse / mondfinsternis' and
     'baptism / taufe' as weak, both of which are correct. So a low score is a
     signal to LOOK, not proof of an error. Every flagged pair is consequently
     listed for manual reading, and the error count below counts only pairs that
     are actually wrong on inspection.

  3. ROUND-TRIP — build an engine whose anchors are the concept's own English
     terms, then ask it with the German term. If the curated mapping is right,
     the German query must route to that concept. This is an end-to-end test of
     the thing the glossary exists for.

Run: uv run python -m evals.curated_glossary_verify
"""
import importlib.util
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402

from klix import Choice, DecisionEngine  # noqa: E402
from klix.backbone import HybridBackbone  # noqa: E402
from klix.glossary import Glossary  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "curated", REPO / "scripts" / "curated_glossary_data.py")
curated = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curated)

WEAK = 0.50   # flag threshold for the agreement check
BAD = 0.35


def main():
    domains = curated.all_domains()

    print("=" * 100)
    print("CURATED GLOSSARY VERIFICATION — all 124 concepts, per domain")
    print("=" * 100)

    bb = HybridBackbone()
    grand = {"n": 0, "structural": 0, "weak": 0, "bad": 0, "roundtrip_fail": 0}
    bad_pairs, weak_pairs, rt_fails = [], [], []

    for domain, mapping in domains.items():
        n = len(mapping)
        # --- 1. structural ---
        findings = Glossary(dict(mapping)).validate()

        # --- 2. cross-lingual agreement ---
        flat = [t for v in mapping.values() for t in v["de"] + v["en"]]
        emb = {}
        for i in range(0, len(flat), 64):
            chunk = flat[i:i + 64]
            for t, vec in zip(chunk, bb.embed_model.embed(chunk)):
                v = np.asarray(vec, dtype=np.float32)
                nn = float(np.linalg.norm(v))
                emb[t] = v / (nn if nn > 0 else 1.0)
        sims, w, b = {}, 0, 0
        for concept, langs in mapping.items():
            best = max((float(emb[d] @ emb[e])
                        for d in langs["de"] for e in langs["en"]), default=0.0)
            sims[concept] = best
            if best < BAD:
                b += 1
                bad_pairs.append((domain, concept, langs["de"], langs["en"], best))
            elif best < WEAK:
                w += 1
                weak_pairs.append((domain, concept, langs["de"], langs["en"], best))

        # --- 3. round trip: English anchors, German query, GLOSSARY APPLIED ---
        # The glossary must be wired in here: a German query against English
        # anchors only stands a chance because the glossary bridges the keyword
        # channel. Testing without it measures the dense channel alone (the
        # documented ~77% baseline) and says nothing about the mapping quality.
        eng = DecisionEngine(glossary=Glossary(dict(mapping)))
        anchors = {c: v["en"] for c, v in mapping.items()}
        eng.add_head(Choice(name="r", options=anchors))
        eng.compile()
        rt = 0
        for concept, langs in mapping.items():
            probe = langs["de"][0]
            got = eng.decide(probe).r
            if got != concept:
                rt += 1
                rt_fails.append((domain, concept, probe, got))

        med = statistics.median(sims.values())
        print(f"\n{domain.upper()}  ({n} concepts, "
              f"{sum(len(v['de'])+len(v['en']) for v in mapping.values())} terms)")
        print(f"  structural findings   : {len(findings)}")
        print(f"  agreement median      : {med:.3f}")
        print(f"  agreement < {BAD} (suspect) : {b}")
        print(f"  agreement {BAD}-{WEAK}       : {w}")
        print(f"  round-trip failures   : {rt}/{n}")

        grand["n"] += n
        grand["structural"] += len(findings)
        grand["weak"] += w
        grand["bad"] += b
        grand["roundtrip_fail"] += rt

    print()
    print("=" * 100)
    print("ALL DOMAINS")
    print("=" * 100)
    print(f"  concepts checked           : {grand['n']}  (all, not a sample)")
    print(f"  structural defects         : {grand['structural']}")
    print(f"  agreement < 0.35 (suspect) : {grand['bad']}")
    print(f"  agreement 0.35-0.50        : {grand['weak']}")
    print(f"  round-trip failures        : {grand['roundtrip_fail']}")

    if bad_pairs:
        print("\n  AGREEMENT < 0.35 — read each and decide error vs. rare-but-correct:")
        for d, c, de, en, s in sorted(bad_pairs, key=lambda x: x[4]):
            print(f"    {s:.3f}  [{d}] {c:20s} de={de}  en={en}")
    if weak_pairs:
        print("\n  AGREEMENT 0.35-0.50:")
        for d, c, de, en, s in sorted(weak_pairs, key=lambda x: x[4]):
            print(f"    {s:.3f}  [{d}] {c:20s} de={de}  en={en}")
    if rt_fails:
        print("\n  ROUND-TRIP FAILURES (German term did NOT route to its own concept):")
        for d, c, probe, got in rt_fails:
            print(f"    [{d}] {c:20s} probe={probe!r} -> got {got!r}")


if __name__ == "__main__":
    main()

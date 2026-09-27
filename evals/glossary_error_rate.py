"""Per-domain error rate of the AUTO layer, sense-based, curated as ground truth.

Why sense-based and not key-based
---------------------------------
Scoring by concept KEY produces false errors: the auto layer resolving
'motor' -> concept 'engine' (en=['engine']) is CORRECT for routing, it just uses
a different key name. Key comparison counted that as wrong.

So the test is: does the auto-resolved concept's ENGLISH side name the same
sense as the curated concept's English side? Measured as the best embedding
cosine between the two English term sets. That is a pairwise "same concept?"
question between two concrete sets — which is where the dense metric is valid.
(It was NOT valid earlier as an absolute quality threshold on the whole file,
where it only measured term rarity and flagged 'baptism/taufe' as bad.)

  correct — cosine >= 0.60  (same concept, possibly a different key)
  wrong   — cosine <  0.60  (different concept -> the harmful case)
  missing — the German term is not in the auto layer at all

Ground truth: the 124 curated concepts, written by hand BEFORE this measurement
was run, so the labels could not be shaped by the scoring.

Run: uv run python -m evals.glossary_error_rate
"""
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402

from klix.backbone import HybridBackbone  # noqa: E402
from klix.glossary import DEFAULT_GLOSSARY, Glossary  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
# The content module imports its sibling `curated_glossary_ext`, so the scripts
# directory must be importable — loading by spec path alone is not enough.
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location(
    "curated", REPO / "scripts" / "curated_glossary_data.py")
curated = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curated)

SAME_SENSE = 0.60


def main():
    auto = Glossary(json.loads(DEFAULT_GLOSSARY.read_text(encoding="utf-8")))
    domains = curated.all_domains()
    bb = HybridBackbone()

    needed = set()
    for mapping in domains.values():
        for langs in mapping.values():
            needed.update(langs["en"])
            for t in langs["de"]:
                k = auto._index.get(t)
                if k:
                    needed.update(auto.mapping[k].get("en", []))
    emb = {}
    todo = sorted(needed)
    for i in range(0, len(todo), 64):
        chunk = todo[i:i + 64]
        for t, v in zip(chunk, bb.embed_model.embed(chunk)):
            v = np.asarray(v, dtype=np.float32)
            n = float(np.linalg.norm(v))
            emb[t] = v / (n if n > 0 else 1.0)

    print("=" * 96)
    print(f"PER-DOMAIN ERROR RATE — sense-based (cosine >= {SAME_SENSE} = same concept)")
    print("=" * 96)

    grand = {"ok": 0, "wrong": 0, "missing": 0, "n": 0}
    for domain, mapping in domains.items():
        ok = wr = mi = n = 0
        wrongs = []
        for concept, langs in mapping.items():
            want_en = langs["en"]
            for term in langs["de"]:
                n += 1
                key = auto._index.get(term)
                if key is None:
                    mi += 1
                    continue
                got_en = auto.mapping[key].get("en", [])
                sim = max((float(emb[w] @ emb[g])
                           for w in want_en for g in got_en
                           if w in emb and g in emb), default=0.0)
                if sim >= SAME_SENSE:
                    ok += 1
                else:
                    wr += 1
                    wrongs.append((term, concept, want_en, key, got_en, sim))
        tested = ok + wr
        rate = wr / tested if tested else 0.0
        print(f"  {domain.upper():14} terms={n:3d}  correct={ok:3d}  wrong={wr:3d}  "
              f"missing={mi:3d}   WRONG-RATE={rate:5.1%}")
        for term, want, wen, key, gen, sim in sorted(wrongs, key=lambda x: x[5])[:8]:
            print(f"      {sim:.2f}  {term:18s} [{want}] {wen}  ->  [{key}] {gen}")
        grand["ok"] += ok
        grand["wrong"] += wr
        grand["missing"] += mi
        grand["n"] += n

    g = grand
    tested = g["ok"] + g["wrong"]
    print()
    print(f"  TOTAL   terms={g['n']}  correct={g['ok']}  wrong={g['wrong']}  missing={g['missing']}")
    print(f"  -> WRONG-MAPPING RATE (where the auto layer had a mapping): {g['wrong']/tested:.1%}")
    print(f"  -> coverage: {tested/g['n']:.1%} of curated German terms exist in the auto layer")
    print()
    print("  'missing' is not an error: the auto layer is a large general vocabulary")
    print("  and does not contain every curated term. 'wrong' is the harmful figure —")
    print("  the term is present and bridges to a different sense.")


if __name__ == "__main__":
    main()

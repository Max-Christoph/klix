"""Unbiased mapping-quality estimate over the WHOLE glossary.

The 20% figure from glossary_correctness.py is measured on a deliberately
demanding, domain-critical probe list (~40 terms chosen as "what a
manufacturing glossary must cover"), and that list is partly failure-enriched.
It is the right denominator for "is this usable for shop-floor tickets", but it
is NOT the error rate of the file as a whole.

This script estimates the whole-file quality objectively, without hand labels:

  For a random sample of concepts, embed the German primary term and the English
  primary terms with the same backbone klix routes with, and measure cosine
  similarity. A CORRECT cross-lingual pair scores high (both sides name the same
  concept); a WRONG mapping (our "mutter -> mam" case) scores low.

  The curated manufacturing preset terms are measured the same way as a
  known-good reference group.

Run: uv run python -m evals.glossary_mapping_quality
"""
import json
import random
import statistics
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402

from klix import manufacturing_glossary, workflow_glossary  # noqa: E402
from klix.backbone import HybridBackbone  # noqa: E402
from klix.glossary import DEFAULT_GLOSSARY  # noqa: E402

N_SAMPLE = 300


def embed_all(bb, texts):
    out = {}
    batch = list(texts)
    for i in range(0, len(batch), 64):
        chunk = batch[i:i + 64]
        for t, v in zip(chunk, bb.embed_model.embed(chunk)):
            v = np.asarray(v, dtype=np.float32)
            n = float(np.linalg.norm(v))
            out[t] = v / (n if n > 0 else 1.0)
    return out


def score_pairs(bb, pairs):
    """pairs: [(de_terms, en_terms)] -> best cosine between the two sides."""
    flat = {t for de, en in pairs for t in de + en}
    emb = embed_all(bb, flat)
    sims = []
    for de, en in pairs:
        best = 0.0
        for d in de:
            for e in en:
                if d in emb and e in emb:
                    best = max(best, float(emb[d] @ emb[e]))
        sims.append(best)
    return sims


def report(name, sims):
    sims = sorted(sims)
    n = len(sims)
    weak = sum(1 for s in sims if s < 0.5)
    bad = sum(1 for s in sims if s < 0.35)
    print(f"  {name:28} n={n:4d}  median {statistics.median(sims):.3f}  "
          f"<0.50 {weak:3d} ({weak/n:5.1%})  <0.35 {bad:3d} ({bad/n:5.1%})")
    return weak / n, bad / n


def main():
    bb = HybridBackbone()
    raw = json.loads(DEFAULT_GLOSSARY.read_text(encoding="utf-8"))

    print("=" * 92)
    print("MAPPING QUALITY — dense cross-lingual agreement (unbiased whole-file estimate)")
    print("=" * 92)
    print("  correct pair -> both sides name the same concept -> high cosine")
    print("  wrong mapping -> low cosine (e.g. 'mutter' [nut] vs 'mam' [mother])")
    print()

    rng = random.Random(20260927)
    sample = rng.sample(sorted(raw), N_SAMPLE)
    gen_pairs = [(raw[k]["de"], raw[k]["en"]) for k in sample]
    w_gen, b_gen = report(f"default glossary (n={N_SAMPLE})", score_pairs(bb, gen_pairs))

    # known-good reference group: the hand-written manufacturing preset
    man = manufacturing_glossary().mapping
    wf = workflow_glossary().mapping
    ref_pairs = [(v["de"], v["en"]) for v in man.values()]
    w_ref, b_ref = report("manufacturing preset (curated)", score_pairs(bb, ref_pairs))
    wf_pairs = [(v["de"], v["en"]) for v in wf.values()]
    w_wf, b_wf = report("workflow preset (curated)", score_pairs(bb, wf_pairs))

    print()
    print("  INTERPRETATION")
    print(f"    curated presets : {w_ref:.1%} / {w_wf:.1%} of pairs below 0.50")
    print(f"    generated file  : {w_gen:.1%} below 0.50, {b_gen:.1%} below 0.35")
    print()
    if w_gen > 2 * max(w_ref, w_wf):
        print("    -> the generated layer is measurably weaker than the curated layer.")
    else:
        print("    -> no large gap between generated and curated layers on this metric.")

    # show the worst offenders, so the numbers are eyeballable
    flat = {t for de, en in gen_pairs for t in de + en}
    emb = embed_all(bb, list(flat))
    scored = []
    for k, (de, en) in zip(sample, gen_pairs):
        best = 0.0
        for d in de:
            for e in en:
                if d in emb and e in emb:
                    best = max(best, float(emb[d] @ emb[e]))
        scored.append((best, k, de, en))
    scored.sort()
    print("\n  20 WORST pairs (lowest cross-lingual agreement):")
    for s, k, de, en in scored[:20]:
        print(f"    {s:.3f}  {k:30s} de={de}  en={en}")


if __name__ == "__main__":
    main()

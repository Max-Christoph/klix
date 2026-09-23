"""Instrumentation: similarity distributions for Choice and Score heads.

Run: uv run python -c "import sys; sys.path.insert(0,'.'); sys.path.insert(0,'src'); import evals.instrument"
"""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402

from klix import DecisionEngine, Choice, Score  # noqa: E402
from evals.eval_baseline import (  # noqa: E402
    CHOICE_OPTIONS, SCORE_LOW, SCORE_HIGH, CHOICE_CASES, SCORE_CASES,
)

eng = DecisionEngine()
eng.add_head(Choice(name="target", options=CHOICE_OPTIONS))
eng.add_head(Score(name="urgency", low_anchors=SCORE_LOW, high_anchors=SCORE_HIGH, min_val=0.0, max_val=3.0))
eng.compile()
heads = {h.name: h for h in eng.heads}


def topk_mean(sims, k=2):
    k = min(k, len(sims))
    return float(np.mean(np.sort(sims)[-k:]))


print("=== CHOICE: best hybrid score per query ===")
for query, expected in CHOICE_CASES:
    encoded = eng.backbone.encode(query)
    dense_sims = heads["target"].dense_matrix @ encoded.dense_vec
    sparse_sims = np.asarray((encoded.sparse_vec @ heads["target"].sparse_matrix.T).todense())[0]
    hybrid = dense_sims + 0.5 * sparse_sims
    tag = f"[{expected}]" if expected else "[OOD]"
    print(f"{tag:12s} best={hybrid.max():.3f} | {query}")

print("\n=== SCORE: max vs top-2-mean per query ===")
for query, lo, hi in SCORE_CASES:
    encoded = eng.backbone.encode(query)
    low_sims = heads["urgency"].low_matrix @ encoded.dense_vec
    high_sims = heads["urgency"].high_matrix @ encoded.dense_vec
    print(
        f"band[{lo}-{hi}] "
        f"low(max={low_sims.max():.3f} top2={topk_mean(low_sims):.3f}) "
        f"high(max={high_sims.max():.3f} top2={topk_mean(high_sims):.3f}) | {query}"
    )
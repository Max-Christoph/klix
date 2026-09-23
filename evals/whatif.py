"""What-if tests: Choice reject-pole + Score top-k & coverage warning (manual simulation)."""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402

from klix import DecisionEngine, Choice, Score  # noqa: E402
from evals.eval_baseline import (  # noqa: E402
    CHOICE_OPTIONS, SCORE_LOW, SCORE_HIGH, CHOICE_CASES, SCORE_CASES,
)

REJECT = ["casual office chat", "birthday wishes", "small talk about the weather", "off topic conversation"]

# --- Idea 1: Choice with reject anchors (manual simulation) -----------------
eng = DecisionEngine()
eng.add_head(Choice(name="target", options=CHOICE_OPTIONS))
eng.compile()
ch = eng.heads[0]

reject_vecs = np.array(list(eng.backbone.embed_model.embed(REJECT)))
reject_vecs = reject_vecs / np.where(
    np.linalg.norm(reject_vecs, axis=1, keepdims=True) == 0, 1.0,
    np.linalg.norm(reject_vecs, axis=1, keepdims=True),
)

print("=== CHOICE with simulated reject pole ===")
for query, expected in CHOICE_CASES:
    encoded = eng.backbone.encode(query)
    dense_sims = ch.dense_matrix @ encoded.dense_vec
    sparse_sims = np.asarray((encoded.sparse_vec @ ch.sparse_matrix.T).todense())[0]
    hybrid = dense_sims + 0.5 * sparse_sims

    cat = {}
    for idx, sim in enumerate(hybrid):
        label = ch.label_map[idx]
        if label not in cat or sim > cat[label]:
            cat[label] = float(sim)
    best_label = max(cat, key=cat.get)
    best_score = cat[best_label]

    reject_sim = float(np.max(reject_vecs @ encoded.dense_vec))
    decision = "REJECT" if reject_sim > best_score else best_label
    tag = f"[{expected}]" if expected else "[OOD]"
    ok = (
        (decision == expected)
        if expected
        else ("ok" if decision == "REJECT" or reject_sim > 0.35 else "WRONG")
    )
    print(f"{tag:12s} best={best_score:.3f} reject={reject_sim:.3f} -> {decision:9s} {ok} | {query}")

# --- Idea 2: Score with top-2 mean + coverage -------------------------------
eng2 = DecisionEngine()
eng2.add_head(Score(name="urgency", low_anchors=SCORE_LOW, high_anchors=SCORE_HIGH, min_val=0.0, max_val=3.0))
eng2.compile()
sh = eng2.heads[0]

def topk_mean(sims, k=2):
    k = min(k, len(sims))
    return float(np.mean(np.sort(sims)[-k:]))

print("\n=== SCORE: top-2 mean + coverage flag (what-if) ===")
for query, lo, hi in SCORE_CASES:
    encoded = eng2.backbone.encode(query)
    s_low = topk_mean(sh.low_matrix @ encoded.dense_vec)
    s_high = topk_mean(sh.high_matrix @ encoded.dense_vec)
    coverage = max(s_low, s_high)
    diff = s_high - s_low
    ratio = 1.0 / (1.0 + np.exp(-diff * 8.0))
    flag = "LOW-COV" if coverage < 0.28 else "       "
    print(f"{flag} cov={coverage:.3f} band[{lo}-{hi}] -> {ratio * 3:.2f} | {query}")
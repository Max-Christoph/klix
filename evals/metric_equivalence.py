"""Verify the claim in point 3: cosine vs euclidean on L2-normalized vectors.

Claim under test: "for L2-normalized embeddings cosine and euclidean are
monotonically equivalent, so switching the metric would change nothing."

Math (both directions):
  For unit vectors ||a|| = ||b|| = 1:
      ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b> = 2 - 2cos(a,b)
  => d_euclid = sqrt(2 - 2*cos)   is a strictly decreasing function of cos
  => ranking by euclid == ranking by cos (reversed), exactly.

So the claim is true IN THE ABSOLUTE RANKING sense. But "changes nothing" needs
a caveat that matters for klix specifically: the HYBRID score adds a sparse
term with a hand-tuned weight, and any *threshold* (reject_threshold,
min_coverage) is calibrated on the similarity SCALE, not the ranking. Both are
scale-dependent. Verify the ranking equivalence empirically and show where the
difference would actually bite.
"""
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402

rng = np.random.default_rng(0)

# 1) algebraic identity on random unit vectors
worst = 0.0
for _ in range(200):
    a = rng.normal(size=64)
    b = rng.normal(size=64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    cos = float(a @ b)
    eu = float(np.linalg.norm(a - b))
    worst = max(worst, abs(eu - np.sqrt(max(2 - 2 * cos, 0.0))))
print(f"1) identity ||a-b|| == sqrt(2-2cos) on unit vectors: max deviation {worst:.2e}")

# 2) ranking equivalence: euclidean argmin == cosine argmax over many trials
mismatch = 0
for _ in range(2000):
    q = rng.normal(size=32)
    q /= np.linalg.norm(q)
    M = rng.normal(size=(20, 32))
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    by_cos = int(np.argmax(M @ q))
    by_eu = int(np.argmin(np.linalg.norm(M - q, axis=1)))
    mismatch += int(by_cos != by_eu)
print(f"2) argmax(cos) vs argmin(euclid) over 2000 trials: {mismatch} mismatches")

# 3) where it would actually bite: the hybrid score mixes scales
print()
print("3) scale sensitivity (why 'changes nothing' is only half true):")
q = rng.normal(size=32)
q /= np.linalg.norm(q)
M = rng.normal(size=(20, 32))
M /= np.linalg.norm(M, axis=1, keepdims=True)
cos = M @ q
eu = np.linalg.norm(M - q, axis=1)
print(f"   cosine  range: [{cos.min():+.4f}, {cos.max():+.4f}]")
print(f"   euclid  range: [{eu.min():.4f}, {eu.max():.4f}]")
print("   -> a keyword_boost of 0.5 added to cosine means something entirely")
print("      different when added to a euclidean distance (different scale AND")
print("      inverted direction). Ranking-equivalent != score-comparable.")
print("   -> every calibrated threshold (reject_threshold, min_coverage,")
print("      the Flag temp softmax) is defined on the cosine scale.")
print()
print("CONCLUSION: point 3 is correct that switching the metric is pointless --")
print("the ranking is provably identical, so no experiment is needed. But the")
print("reason is stronger than 'monotone equivalent': euclidean is an exact")
print("affine-in-cos reparametrization, and klix's hybrid score + thresholds are")
print("built on the cosine scale, so a switch would require recalibrating all of")
print("them for zero ranking benefit.")

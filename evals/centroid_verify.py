"""Verification round for the P1/P2 findings.

Two things the first pass left open:

  (a) My "centroid" was the MEAN OF SIMILARITY SCORES per label. The standard
      reading of "centroid" is cosine to the MEAN VECTOR of the class. These
      are NOT the same:
          mean_i cos(q, m_i)  !=  cos(q, mean_i m_i)
      Test both, plus the per-domain breakdown, so we know what the +8.6pt on
      cross-domain actually buys and where it comes from.

  (b) The cross-domain centroid gain (+8.6pt, n=70) sits right at the 95% CI
      (±9pt). Bootstrap the difference (centroid - nearest) to see whether it
      survives resampling, instead of declaring victory on a point estimate.

RRF is already conclusively negative (-14 to -25pt on all three corpora) and is
not re-run here.

Run: uv run python -m evals.centroid_verify
"""
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402

from klix import Choice, DecisionEngine  # noqa: E402
from evals.benchmark import DATASETS  # noqa: E402
from evals.benchmark_bilingual import OPTIONS as BI_OPT, TEST_DE, TEST_EN  # noqa: E402
from evals.corpus_expansion import build_expanded_corpus  # noqa: E402
from evals.eval_domains import IMG_CASES, IMG_OPTIONS, SHOP_CASES, SHOP_OPTIONS, TASK_CASES, TASK_OPTIONS  # noqa: E402
from evals.variant_sweep import FIN_OPTIONS, FIN_TESTS, HR_OPTIONS, HR_TESTS  # noqa: E402

SETS = [
    ("HR", HR_OPTIONS, HR_TESTS),
    ("FIN", FIN_OPTIONS, FIN_TESTS),
    ("IMAGE", IMG_OPTIONS, [(q, e) for q, e in IMG_CASES if e]),
    ("TASK", TASK_OPTIONS, [(q, e) for q, e in TASK_CASES if e]),
    ("SHOP", SHOP_OPTIONS, [(q, e) for q, e in SHOP_CASES if e]),
]
CORPUS = build_expanded_corpus(SETS)
BI_GROUPS = [("BI", BI_OPT, [(t, e) for t, e in TEST_EN if e] + [(t, e) for t, e in TEST_DE if e])]
XD_GROUPS = [(n, o, [(t, e) for t, e in ts if e]) for n, o, ts in DATASETS]
EXP_GROUPS = [(n, o, CORPUS["domains"][n]["cases"]) for n, o, _ in SETS]


def _finish(cats):
    best = max(cats, key=cats.get)
    srt = sorted(cats.values(), reverse=True)
    run = srt[1] if len(srt) > 1 else 0.0
    conf = float(np.clip((cats[best] - run) / (cats[best] + 1e-5) * 1.5, 0.0, 1.0))
    return {"value": best, "score": cats[best], "confidence": conf, "scores": cats}


def make_head(mode: str, options, topk: int | None = None, **kw):
    """mode: nearest | simmean (mean of similarities) | cenvec (cosine to mean vector)."""
    head = Choice(name="h", options=options, **kw)
    if mode == "nearest":
        return head

    if mode == "simmean":
        def evaluate(encoded, _h=head, _t=topk):
            d = _h.dense_matrix @ encoded.dense_vec
            s = (np.zeros_like(d) if encoded.sparse_vec is None
                 else np.asarray(_h.sparse_matrix @ np.asarray(encoded.sparse_vec.todense()).ravel()).ravel())
            hyb = d + _h.keyword_boost * s
            cats = {}
            for lab, rows in _h._label_rows.items():
                sims = hyb[rows]
                cats[lab] = (float(np.mean(sims[np.argsort(sims)[-min(_t, len(sims)):]]))
                             if _t else float(np.mean(sims)))
            return _finish(cats)
        head.evaluate = evaluate
        return head

    if mode == "cenvec":
        # true centroid: normalize the mean anchor vector per label, then cosine.
        # Also fuse with the sparse channel via the same keyword_boost weight.
        def evaluate(encoded, _h=head, _t=topk):
            cen = {}
            for lab, rows in _h._label_rows.items():
                sub = _h.dense_matrix[rows]
                if _t:  # centroid of the best-k anchors for this query
                    sims = sub @ encoded.dense_vec
                    idx = np.argsort(sims)[-min(_t, len(sims)):]
                    sub = sub[idx]
                v = sub.mean(axis=0)
                n = np.linalg.norm(v)
                cen[lab] = v / n if n > 0 else v
            C = np.vstack([cen[lab] for lab in _h._label_rows])
            d = C @ encoded.dense_vec
            s = (np.zeros_like(d) if encoded.sparse_vec is None
                 else np.asarray(_h.sparse_matrix @ np.asarray(encoded.sparse_vec.todense()).ravel()).ravel())
            # sparse per-label mean, aligned to the same label order
            sp = {lab: float(np.mean(s[rows])) for lab, rows in _h._label_rows.items()}
            cats = {lab: float(d[i]) + _h.keyword_boost * sp[lab]
                    for i, lab in enumerate(_h._label_rows)}
            return _finish(cats)
        head.evaluate = evaluate
        return head

    raise ValueError(mode)


def per_domain(groups, mode, topk=None):
    """Returns list of (name, correct, total, texts, expected, preds)."""
    out = []
    for name, options, tests in groups:
        eng = DecisionEngine()
        eng.add_head(make_head(mode, options, topk=topk))
        eng.compile()
        preds = [eng.decide(t).h for t, _ in tests]
        exp = [e for _, e in tests]
        out.append((name, sum(p == e for p, e in zip(preds, exp)), len(tests), preds, exp))
    return out


def bootstrap_diff(groups, mode, topk=None, n_boot=2000, seed=7):
    """Bootstrap 95% CI for (centroid - nearest) accuracy difference, pooled."""
    base = per_domain(groups, "nearest")
    cand = per_domain(groups, mode, topk)
    correct_b, correct_c, n = [], [], 0
    for (bn, bok, btot, bp, be), (_cn, cok, ctot, cp, ce) in zip(base, cand):
        correct_b.extend([int(p == e) for p, e in zip(bp, be)])
        correct_c.extend([int(p == e) for p, e in zip(cp, ce)])
    correct_b, correct_c = np.array(correct_b), np.array(correct_c)
    n = len(correct_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(correct_c[idx].mean() - correct_b[idx].mean())
    diffs = np.sort(diffs)
    return (correct_b.mean(), correct_c.mean(),
            diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)], n)


def report(corpus_name, groups, modes):
    print(f"\n================ {corpus_name} ================")
    for label, mode, topk in modes:
        rows = per_domain(groups, mode, topk)
        ok = sum(r[1] for r in rows)
        tot = sum(r[2] for r in rows)
        detail = "  ".join(f"{r[0]} {r[1]}/{r[2]}" for r in rows)
        print(f"  {label:34s} {ok:3d}/{tot} = {ok/tot:6.1%}")
        print(f"    {'':34s} {detail}")


M1 = [("nearest (baseline)", "nearest", None),
      ("nearest + topk2", "nearest", None)]
report("BILINGUAL (20 cases)", BI_GROUPS, [
    ("nearest (baseline)", "nearest", None),
    ("simmean: mean of similarities", "simmean", None),
    ("simmean of best-2", "simmean", 2),
    ("cenvec: cosine to mean vector", "cenvec", None),
    ("cenvec of best-2", "cenvec", 2),
    ("cenvec of best-3", "cenvec", 3),
])

report("CROSS-DOMAIN (70 cases)", XD_GROUPS, [
    ("nearest (baseline)", "nearest", None),
    ("simmean: mean of similarities", "simmean", None),
    ("simmean of best-3", "simmean", 3),
    ("cenvec: cosine to mean vector", "cenvec", None),
    ("cenvec of best-2", "cenvec", 2),
    ("cenvec of best-3", "cenvec", 3),
])

report("EXPANDED (273 cases)", EXP_GROUPS, [
    ("nearest (baseline)", "nearest", None),
    ("simmean: mean of similarities", "simmean", None),
    ("cenvec: cosine to mean vector", "cenvec", None),
    ("cenvec of best-3", "cenvec", 3),
])

print("\n================ BOOTSTRAP: centroid minus nearest ================")
print("(95% CI from 2000 resamples; if the interval contains 0, the gain is not")
print(" distinguishable from noise at this n)")
for cname, groups, mode, topk in (
    ("bilingual simmean", BI_GROUPS, "simmean", None),
    ("cross simmean", XD_GROUPS, "simmean", None),
    ("cross cenvec best-3", XD_GROUPS, "cenvec", 3),
    ("cross simmean best-3", XD_GROUPS, "simmean", 3),
    ("expanded simmean", EXP_GROUPS, "simmean", None),
    ("expanded cenvec best-3", EXP_GROUPS, "cenvec", 3),
):
    try:
        b, c, lo, hi, n = bootstrap_diff(groups, mode, topk)
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not distinguishable"
        print(f"  {cname:24s} nearest {b:5.1%} -> centroid {c:5.1%}   "
              f"diff {c-b:+.1%}  CI95 [{lo:+.1%}, {hi:+.1%}]  n={n}  {sig}")
    except Exception as ex:  # noqa: BLE001
        print(f"  {cname:24s} FAILED {type(ex).__name__}: {str(ex)[:40]}")

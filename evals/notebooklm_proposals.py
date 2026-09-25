"""Isolated experiments for the NotebookLM-derived proposals.

Each variant is additive and reversible; nothing here changes defaults.
Everything is measured against the CURRENT state on the repo's own corpora.

  P1  Centroid head: score the query against the MEAN of all anchors per class
      instead of the single nearest anchor (plus a centroid-of-best-k variant).

  P2  RRF fusion: replace the weighted dense+sparse score addition
      (keyword_boost) with Reciprocal Rank Fusion -- rank each channel
      separately, fuse by rank (k=40 and k=60 tested).

Corpora (all from this repo, no new data invented):
  bilingual    20 cases, mixed DE/EN anchors
  cross-domain 70 cases, 6 domains
  expanded     273 cases (60 frozen + generated anchor paraphrases)

Reference points from earlier runs:
  bilingual : nearest 15/20, topk2 16/20, linear 15/20
  cross-dom : nearest 72%, linear 84%
  expanded  : nearest 93.0%, linear 96.7%

Run: uv run python -m evals.notebooklm_proposals
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


def _channels(head, encoded):
    """Dense and sparse similarity vectors over all anchors."""
    dense = head.dense_matrix @ encoded.dense_vec
    if encoded.sparse_vec is None:
        sparse = np.zeros_like(dense)
    else:
        q = np.asarray(encoded.sparse_vec.todense()).ravel()
        sparse = np.asarray(head.sparse_matrix @ q).ravel()
    return dense, sparse


def _finish(cats: dict) -> dict:
    best = max(cats, key=cats.get)
    srt = sorted(cats.values(), reverse=True)
    run = srt[1] if len(srt) > 1 else 0.0
    conf = float(np.clip((cats[best] - run) / (cats[best] + 1e-5) * 1.5, 0.0, 1.0))
    return {"value": best, "score": cats[best], "confidence": conf, "scores": cats}


def make_head(mode: str, options, topk: int | None = None, rrf_k: int = 60, **kw):
    """mode: 'nearest' | 'centroid' | 'rrf'. Library code untouched."""
    head = Choice(name="h", options=options, **kw)

    if mode == "nearest":
        return head

    if mode == "centroid":
        def evaluate(encoded, _h=head, _t=topk):
            dense, sparse = _channels(_h, encoded)
            hybrid = dense + _h.keyword_boost * sparse
            cats = {}
            for lab, rows in _h._label_rows.items():
                sims = hybrid[rows]
                cats[lab] = (float(np.mean(sims[np.argsort(sims)[-min(_t, len(sims)):]]))
                             if _t else float(np.mean(sims)))
            return _finish(cats)
        head.evaluate = evaluate
        return head

    if mode == "rrf":
        def evaluate(encoded, _h=head, _k=rrf_k, _t=topk):
            dense, sparse = _channels(_h, encoded)

            def ranks(sim):
                order = np.argsort(-sim)
                r = np.empty(len(sim), dtype=float)
                r[order] = np.arange(1, len(sim) + 1)
                return r

            fused = 1.0 / (_k + ranks(dense)) + 1.0 / (_k + ranks(sparse))
            cats = {}
            for lab, rows in _h._label_rows.items():
                sims = fused[rows]
                cats[lab] = (float(np.mean(sims[np.argsort(sims)[-min(_t, len(sims)):]]))
                             if _t else float(np.max(sims)))
            return _finish(cats)
        head.evaluate = evaluate
        return head

    raise ValueError(mode)


# ---------------------------------------------------------------------------
# corpora
# ---------------------------------------------------------------------------
BI_CASES = [(t, e) for t, e in TEST_EN if e] + [(t, e) for t, e in TEST_DE if e]
XD_SETS = [(n, o, [(t, e) for t, e in ts if e]) for n, o, ts in DATASETS]


def measure(corpus: str, mode: str, topk: int | None = None, rrf_k: int = 60, **kw):
    if corpus == "bilingual":
        groups = [("BI", BI_OPT, BI_CASES)]
    elif corpus == "cross":
        groups = XD_SETS
    elif corpus == "expanded":
        groups = [(n, o, CORPUS["domains"][n]["cases"]) for n, o, _ in SETS]
    else:
        raise ValueError(corpus)

    ok = tot = 0
    for _name, options, tests in groups:
        eng = DecisionEngine()
        eng.add_head(make_head(mode, options, topk=topk, rrf_k=rrf_k, **kw))
        eng.compile()
        ok += sum(1 for t, e in tests if eng.decide(t).h == e)
        tot += len(tests)
    return ok, tot


def show(corpus: str, label: str, **kw):
    try:
        ok, tot = measure(corpus, **kw)
        print(f"  {label:40s} {ok:3d}/{tot} = {ok/tot:6.1%}")
    except Exception as ex:  # noqa: BLE001
        print(f"  {label:40s} FAILED {type(ex).__name__}: {str(ex)[:42]}")


print()
print("################ P1: CENTROID vs NEAREST ################")
for corpus in ("bilingual", "cross", "expanded"):
    print(f"\n-- {corpus} --")
    show(corpus, "nearest (baseline)", mode="nearest")
    show(corpus, "nearest + topk2 (baseline)", mode="nearest", label_aggregation="topk")
    show(corpus, "CENTROID (mean of all anchors)", mode="centroid")
    show(corpus, "CENTROID of best-2 anchors", mode="centroid", topk=2)
    show(corpus, "CENTROID of best-3 anchors", mode="centroid", topk=3)

print()
print("################ P2: RRF vs keyword_boost ################")
for corpus in ("bilingual", "cross", "expanded"):
    print(f"\n-- {corpus} --")
    show(corpus, "keyword_boost (baseline)", mode="nearest")
    show(corpus, "RRF k=60", mode="rrf", rrf_k=60)
    show(corpus, "RRF k=40", mode="rrf", rrf_k=40)
    show(corpus, "RRF k=60 + topk2", mode="rrf", rrf_k=60, topk=2)

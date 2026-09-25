"""Round-2 experiments (P1: squared-euclidean centroid scoring).

MEASURED RESULTS (2026-09-25, dev workstation)
----------------------------------------------
Centroid + squared euclidean vs centroid + cosine, bootstrap 3000 resamples:

    corpus      cosine   l2sq    diff     95% CI              verdict
    bilingual   75.0%    70.0%   -5.0%   [-15.0%, +0.0%]     not distinguishable
    cross       84.3%    85.7%   +1.4%   [ -2.9%, +5.7%]     not distinguishable
    expanded    96.7%    97.1%   +0.4%   [ -0.7%, +1.5%]     not distinguishable
    60 frozen   85.0%    86.7%   +1.7%   (single case)        not distinguishable

Control arm: squared euclidean on the UNIT-normalized centroid (`l2sq_norm`)
reproduces cosine EXACTLY on cross and expanded (identical per-item
predictions) and differs by one case on bilingual. That confirms the algebra --
the arms differ only by the centroid-norm term ( -||q-c||^2 = -1 - ||c||^2 +
2<q,c> vs <q,c>/||c|| ), and that term changes nothing measurable at these
anchor counts. The hypothesis about the norm carrying variance information is
mathematically correct but below the noise floor here.
-> NOT adopted. No parameter added.

Run: uv run python -m evals.round2_p1 <corpus> <mode>
     corpus: bilingual | cross | expanded | frozen
     mode:   nearest | cos | l2sq | l2sq_norm

Each run persists per-item correctness to evals/_p1_results/ so a later run can
bootstrap the difference.
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

GROUPS = {
    "bilingual": [("BI", BI_OPT, [(t, e) for t, e in TEST_EN if e] + [(t, e) for t, e in TEST_DE if e])],
    "cross": [(n, o, [(t, e) for t, e in ts if e]) for n, o, ts in DATASETS],
    "expanded": [(n, o, CORPUS["domains"][n]["cases"]) for n, o, _ in SETS],
    "frozen": [(n, o, [(q, e) for q, e in ts if e]) for n, o, ts in SETS],
}


def make(mode: str, options: dict):
    if mode == "nearest":
        return Choice(name="h", options=options)
    head = Choice(name="h", options=options, classifier="centroid")
    if mode == "cos":
        return head

    def evaluate(encoded, _h=head, _m=mode):
        cents = np.vstack([_h.dense_matrix[rows].mean(axis=0)
                           for rows in _h._label_rows.values()])
        if _m == "l2sq_norm":
            n = np.linalg.norm(cents, axis=1, keepdims=True)
            cents = cents / np.where(n == 0, 1.0, n)
        sq = -np.sum((cents - encoded.dense_vec) ** 2, axis=1)
        cats = {lab: float(sq[i]) for i, lab in enumerate(_h._label_rows)}
        best = max(cats, key=cats.get)
        srt = sorted(cats.values(), reverse=True)
        run = srt[1] if len(srt) > 1 else 0.0
        return {"value": best, "score": cats[best], "confidence": 0.0, "scores": cats}

    head.evaluate = evaluate
    return head


def main(corpus: str, mode: str) -> None:
    import os

    ok = tot = 0
    preds, exp_all = [], []
    for _n, options, tests in GROUPS[corpus]:
        eng = DecisionEngine()
        eng.add_head(make(mode, options))
        eng.compile()
        for t, e in tests:
            got = eng.decide(t).h
            preds.append(got)
            exp_all.append(e)
            tot += 1
            ok += int(got == e)
    print(f"{corpus:10s} {mode:10s} {ok:3d}/{tot} = {ok/tot:6.1%}")

    out = os.path.join("evals", "_p1_results")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{corpus}_{mode}.txt"), "w", encoding="utf-8") as fh:
        fh.write("".join("1" if p == e else "0" for p, e in zip(preds, exp_all)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "frozen",
         sys.argv[2] if len(sys.argv) > 2 else "cos")

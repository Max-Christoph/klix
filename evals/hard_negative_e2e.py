"""D.3 end-to-end verification with HOLDOUT split (leakage-free).

Methodology (reviewed 2026-09-24, v2): v1 measured before/after on the same
frozen 60 the mining saw (memorization, not generalization). v2 used the
generated variants as eval — but those are anchor paraphrases at ~100%
baseline accuracy, too easy to show any effect (+0 delta).

v3 design (this file):
- MINE/REVIEW: 8 of 12 frozen cases per domain (baseline errors here drive
  the counterexamples).
- EVAL: the 4 HELD-OUT frozen cases per domain — real adversarial cases the
  mining phase never observed. n_eval = 20 across 5 domains; small, so the
  report carries the caveat and per-case detail instead of a bare percent.
- Baseline for the eval set is computed on the SAME holdout with an engine
  that never saw any mining, so the delta isolates the counterexample effect.

Honest expectation: if counterexamples only memorize mining-set mistakes,
holdout delta = +0. A positive delta on held-out cases is evidence that
mined confusions generalize to unseen wording of the same classes.
"""
import os
import tempfile

from linear_sweep import HR_OPTIONS, HR_TESTS, FIN_OPTIONS, FIN_TESTS, IMG_OPTIONS, IMG_CASES, TASK_OPTIONS, TASK_CASES, SHOP_OPTIONS, SHOP_CASES

from klix import DecisionEngine, Choice, HardNegativeStore, attach_counterexamples


def build(options: dict) -> DecisionEngine:
    eng = DecisionEngine()
    eng.add_head(Choice(name="h", options=options))
    eng.compile()
    return eng


def run(eng: DecisionEngine, tests: list, store: HardNegativeStore | None = None) -> tuple[int, int]:
    ok = 0
    for ticket, expected in tests:
        res = eng.decide(ticket)
        if store is not None:
            store.observe(res)
        if res.h == expected:
            ok += 1
    return ok, len(tests)


sets = [
    ("HR", HR_OPTIONS, HR_TESTS),
    ("FIN", FIN_OPTIONS, FIN_TESTS),
    ("IMAGE", IMG_OPTIONS, [(q, e) for q, e in IMG_CASES if e]),
    ("TASK", TASK_OPTIONS, [(q, e) for q, e in TASK_CASES if e]),
    ("SHOP", SHOP_OPTIONS, [(q, e) for q, e in SHOP_CASES if e]),
]

tmpdir = tempfile.mkdtemp()
print("=" * 100)
print("D.3 E2E HOLDOUT — mine on 8 frozen cases/domain, evaluate on 4 HELD-OUT cases/domain")
print("=" * 100)
print(f"{'DATASET':8s} {'mined':>6s} {'reviewed':>9s} {'holdout-before':>15s} {'holdout-after':>14s} {'delta':>6s}")
print("-" * 100)
tot_b, tot_a, tot_n = 0, 0, 0
detail_log = []
for name, options, tests in sets:
    mine_set, eval_set = tests[:8], tests[8:]  # deterministic holdout, same for both rounds

    # Mining phase: baseline on the MINE split only.
    eng1 = build(options)
    path = os.path.join(tmpdir, f"{name.lower()}.jsonl")
    store = HardNegativeStore(path=path)
    _baseline_ok, _ = run(eng1, mine_set, store)
    mined = store.mine("h", min_margin=0.05)

    expected_map = dict(mine_set)
    reviewed = [c for c in mined
                if expected_map.get(c["text"]) not in (None, c["picked"])]

    # Evaluation phase on HELD-OUT cases (real adversarial cases).
    before_ok, _ = run(build(options), eval_set)
    eng_after = build(options)
    attach_counterexamples(eng_after.heads[0], reviewed)
    eng_after.compile()
    after_ok, _ = run(eng_after, eval_set)

    tot_b += before_ok
    tot_a += after_ok
    tot_n += len(eval_set)
    print(f"{name:8s} {len(mined):>6d} {len(reviewed):>9d} "
          f"{before_ok:>9d}/{len(eval_set):<3d} {after_ok:>8d}/{len(eval_set):<3d} {after_ok-before_ok:>+5d}")

print("-" * 100)
print(f"HOLDOUT TOTAL: before {tot_b}/{tot_n} = {tot_b/tot_n:.1%}  "
      f"after {tot_a}/{tot_n} = {tot_a/tot_n:.1%}")
print("Caveat: n_eval=20 — small. Delta direction is meaningful, exact size is not.")
print("Positive delta on held-out cases = mined confusions generalize beyond memorized mistakes.")
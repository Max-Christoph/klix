"""D.3 end-to-end verification: hard-negative mining closes the loop.

Simulated live traffic -> HardNegativeStore -> mine() -> add_counterexamples()
-> recompile -> does accuracy on the SAME unchanged cases improve?

The honest setup: mine on the training-free nearest path (default knobs),
feed the wrongly-pulled label as counterexample, recompile, re-measure.
"""
import tempfile
import os

from linear_sweep import HR_OPTIONS, HR_TESTS, FIN_OPTIONS, FIN_TESTS, IMG_OPTIONS, IMG_CASES, TASK_OPTIONS, TASK_CASES, SHOP_OPTIONS, SHOP_CASES

from klix import DecisionEngine, Choice, HardNegativeStore, attach_counterexamples


def build(options: dict) -> tuple[DecisionEngine, Choice]:
    eng = DecisionEngine()
    head = Choice(name="h", options=options)
    eng.add_head(head)
    eng.compile()
    return eng, head


def run_cases(eng: DecisionEngine, tests: list, store: HardNegativeStore) -> tuple[int, int]:
    ok = 0
    for ticket, expected in tests:
        res = eng.decide(ticket)
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
print("=" * 88)
print("D.3 E2E: baseline -> mine hard negatives from its own mistakes -> counterexamples -> recompile")
print("=" * 88)
print(f"{'DATASET':8s} {'before':>10s} {'mined':>6s} {'applied':>8s} {'after':>10s} {'delta':>7s}")
print("-" * 88)
total_before, total_after, total_n = 0, 0, 0
for name, options, tests in sets:
    # Round 1: baseline + collect "live traffic" with the store.
    eng1, head1 = build(options)
    path = os.path.join(tmpdir, f"{name.lower()}.jsonl")
    store = HardNegativeStore(path=path)
    ok1, n = run_cases(eng1, tests, store)
    mined = store.mine("h", min_margin=0.05)

    # Human-in-the-loop step (simulated): each mined case's "picked" label is
    # what the engine wrongly pulled toward -> attach as counterexample.
    # Ground truth: the case text's real label from the test set. A case is a
    # TRUE hard negative when picked != expected.
    expected_map = dict(tests)
    reviewed = []
    for case in mined:
        exp = expected_map.get(case["text"])
        if exp is not None and exp != case["picked"]:
            reviewed.append(case)
    eng2, head2 = build(options)
    applied = attach_counterexamples(head2, reviewed)
    eng2.compile()
    store2 = HardNegativeStore()  # no persistence for round 2
    ok2, _ = run_cases(eng2, tests, store2)

    total_before += ok1
    total_after += ok2
    total_n += n
    print(f"{name:8s} {ok1:>7d}/{n:<2d} {len(mined):>6d} {len(reviewed):>8d} "
          f"{ok2:>7d}/{n:<2d} {ok2-ok1:>+5d}")

print("-" * 88)
print(f"TOTAL before {total_before}/{total_n} = {total_before/total_n:.1%}  "
      f"after {total_after}/{total_n} = {total_after/total_n:.1%}")
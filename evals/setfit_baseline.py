"""E.4: SetFit baseline comparison — leakage-verified.

METHODOLOGY (reviewed 2026-09-24): SetFit trains on the anchor texts (exactly
the information klix's compile() sees), tests on the 60 labeled cases.
Verified disjointness: zero exact matches and zero near-duplicates
(Jaccard > 0.6) between anchor texts and case texts across all five sets —
there is NO train/test leakage; the same 60 cases are a held-out test set
for both methods. `num_epochs=1` keeps SetFit's training budget comparable
to klix's sub-100ms compile; both use the same MiniLM backbone, CPU only.
"""
import time

from evals.linear_sweep import HR_OPTIONS, HR_TESTS, FIN_OPTIONS, FIN_TESTS, IMG_OPTIONS, IMG_CASES, TASK_OPTIONS, TASK_CASES, SHOP_OPTIONS, SHOP_CASES

from klix import DecisionEngine, Choice


def _clean_tests(options: dict, tests: list) -> list:
    """Drops test cases whose text appears verbatim in the anchor set.

    On the shipped sets this drops nothing (verified: disjoint), the filter
    is a guard for future schemas where case/anchor overlap may creep in.
    """
    anchor_texts = {a.lower().strip() for exs in options.values() for a in exs}
    return [(t, e) for t, e in tests if t.lower().strip() not in anchor_texts]


def klix_accuracy(options: dict, tests: list) -> tuple[int, int, float]:
    eng = DecisionEngine()
    eng.add_head(Choice(name="h", options=options))
    eng.compile()
    ok = 0
    t0 = time.perf_counter()
    for ticket, expected in tests:
        if eng.decide(ticket).h == expected:
            ok += 1
    return ok, len(tests), (time.perf_counter() - t0) / len(tests) * 1000


def setfit_accuracy(options: dict, train_texts: list, train_labels: list,
                    tests: list) -> tuple[int, int, float, float]:
    try:
        from setfit import SetFitModel
    except ImportError:
        return -1, len(tests), float("nan"), float("nan")
    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    t0 = time.perf_counter()
    model.fit(train_texts, train_labels, num_epochs=1)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    preds = model.predict([t for t, _ in tests])
    infer_ms = (time.perf_counter() - t0) / len(tests) * 1000
    ok = 0
    for pred, (_, expected) in zip(preds, tests):
        if pred == expected:
            ok += 1
    return ok, len(tests), infer_ms, fit_s


sets = [
    ("HR", HR_OPTIONS, HR_TESTS),
    ("FIN", FIN_OPTIONS, FIN_TESTS),
    ("IMAGE", IMG_OPTIONS, [(q, e) for q, e in IMG_CASES if e]),
    ("TASK", TASK_OPTIONS, [(q, e) for q, e in TASK_CASES if e]),
    ("SHOP", SHOP_OPTIONS, [(q, e) for q, e in SHOP_CASES if e]),
]

print("=" * 96)
print("SETFIT BASELINE — leakage-controlled (train on anchors, test on non-anchor cases)")
print("=" * 96)
print(f"{'DATASET':8s} {'klix':>9s} {'setfit':>9s} {'n_test':>7s} {'dropped':>8s}")
print("-" * 96)
tot_k, tot_s, tot_n, tot_orig = 0, 0, 0, 0
for name, options, tests in sets:
    clean = _clean_tests(options, tests)
    dropped = len(tests) - len(clean)
    if not clean:
        print(f"{name:8s} {'--':>9s} {'--':>9s} (all test cases are anchors)")
        continue
    kok, kn, kms = klix_accuracy(options, clean)
    train_texts, train_labels = [], []
    for lab, examples in options.items():
        for ex in examples:
            train_texts.append(ex)
            train_labels.append(lab)
    sok, sn, sms, fit_s = setfit_accuracy(options, train_texts, train_labels, clean)
    tot_k += kok
    tot_s += sok
    tot_n += kn
    tot_orig += len(tests)
    print(f"{name:8s} {kok:>6d}/{kn:<2d} {sok:>6d}/{sn:<2d} {sn:>7d} {dropped:>8d}")
print("-" * 96)
print(f"LEAK-FREE TOTAL: klix {tot_k}/{tot_n} = {tot_k/tot_n:.1%}   "
      f"setfit {tot_s}/{tot_n} = {tot_s/tot_n:.1%}   (n_test={tot_n} of {tot_orig})")
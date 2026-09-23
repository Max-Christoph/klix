"""E.4: honest baseline comparison against SetFit (few-shot training).

SetFit trains a sentence-transformers fine-tune on few labeled examples —
comparable setup cost to writing anchors (both need ~8-20 examples per class).
This benchmark answers honestly: does klix's training-free approach actually
match trained few-shot classification at the same example budget?

Measured on the SAME unchanged labeled cases as all other evals. CPU-only.
"""
import time

from linear_sweep import HR_OPTIONS, HR_TESTS, FIN_OPTIONS, FIN_TESTS, IMG_OPTIONS, IMG_CASES, TASK_OPTIONS, TASK_CASES, SHOP_OPTIONS, SHOP_CASES

from klix import DecisionEngine, Choice


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


def setfit_accuracy(options: dict, tests: list) -> tuple[int, int, float]:
    try:
        from setfit import SetFitModel
    except ImportError:
        return -1, len(tests), float("nan")
    texts, labels = [], []
    label_names = list(options)
    for lab, examples in options.items():
        for ex in examples:
            texts.append(ex)
            labels.append(lab)
    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    t0 = time.perf_counter()
    model.fit(texts, labels, num_epochs=1)
    fit_s = time.perf_counter() - t0
    ok = 0
    t0 = time.perf_counter()
    preds = model.predict([t for t, _ in tests])
    infer_ms = (time.perf_counter() - t0) / len(tests) * 1000
    for pred, (_, expected) in zip(preds, tests):
        if pred == expected:
            ok += 1
    return ok, len(tests), infer_ms


sets = [
    ("HR", HR_OPTIONS, HR_TESTS),
    ("FIN", FIN_OPTIONS, FIN_TESTS),
    ("IMAGE", IMG_OPTIONS, [(q, e) for q, e in IMG_CASES if e]),
    ("TASK", TASK_OPTIONS, [(q, e) for q, e in TASK_CASES if e]),
    ("SHOP", SHOP_OPTIONS, [(q, e) for q, e in SHOP_CASES if e]),
]

print("=" * 84)
print("KLIX (training-free anchors) vs SETFIT (few-shot training, same examples)")
print("=" * 84)
print(f"{'DATASET':8s} {'klix':>10s} {'klix ms':>9s} {'setfit':>10s} {'setfit ms':>10s}")
print("-" * 84)
tot_k, tot_s, tot_n = 0, 0, 0
for name, options, tests in sets:
    kok, kn, kms = klix_accuracy(options, tests)
    sok, sn, sms = setfit_accuracy(options, tests)
    tot_k += kok
    tot_s += max(0, sok)
    tot_n += kn
    if sok < 0:
        print(f"{name:8s} {kok:>7d}/{kn:<2d} {kms:>9.1f} {'skipped':>10s}")
        continue
    print(f"{name:8s} {kok:>7d}/{kn:<2d} {kms:>9.1f} {sok:>7d}/{sn:<2d} {sms:>10.1f}")
print("-" * 84)
print(f"TOTAL klix {tot_k}/{tot_n} = {tot_k/tot_n:.1%}   setfit {tot_s}/{tot_n} = {tot_s/tot_n:.1%}")
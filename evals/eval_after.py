"""After-measurement: same labeled cases, now using the new head features.

Run: uv run python -c "import sys; sys.path.insert(0,'.'); sys.path.insert(0,'src'); import evals.eval_after"
"""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from klix import DecisionEngine, Choice, Score, Flag  # noqa: E402
from evals.eval_baseline import (  # noqa: E402
    CHOICE_OPTIONS, SCORE_LOW, SCORE_HIGH, FLAG_TRUE, FLAG_FALSE, FLAG_NEUTRAL,
    CHOICE_CASES, SCORE_CASES, FLAG_CASES,
)

REJECT = ["casual office chat", "birthday wishes", "small talk about the weather", "off topic conversation"]

eng = DecisionEngine()
eng.add_head(Choice(name="target", options=CHOICE_OPTIONS, reject_anchors=REJECT))
eng.add_head(Score(name="urgency", low_anchors=SCORE_LOW, high_anchors=SCORE_HIGH,
                   min_val=0.0, max_val=3.0, aggregation="topk"))
eng.add_head(Flag(name="is_security",
                  true_anchors=FLAG_TRUE, false_anchors=FLAG_FALSE,
                  neutral_anchors=FLAG_NEUTRAL, threshold=0.5))
eng.compile()

print("=" * 70)
print("AFTER (reject pole + topk aggregation + coverage)")
print("=" * 70)

# Choice
choice_ok = 0
ood_reject = 0
ood_total = 0
errors = []
for query, expected in CHOICE_CASES:
    d = eng.decide(query).details("target")
    tag = f"[{expected}]" if expected else "[OOD]"
    if expected is None:
        ood_total += 1
        if d["value"] is None:
            ood_reject += 1
        else:
            errors.append((tag, d["score"], d["reject_score"], d["value"], query))
        print(f"{tag:12s} -> {str(d['value']):9s} (reject={d['reject_score']:.3f}) | {query}")
    elif d["value"] == expected:
        choice_ok += 1
        print(f"{tag:12s} -> {d['value']:9s} OK")
    else:
        errors.append((tag, d["score"], d["reject_score"], d["value"], query))
        print(f"{tag:12s} -> {d['value']:9s} ERROR (expected {expected})")

n_in = len([c for c in CHOICE_CASES if c[1] is not None])
print(f"\nChoice accuracy: {choice_ok}/{n_in} = {choice_ok / n_in:.0%}")
print(f"OOD rejection:   {ood_reject}/{ood_total}")
if errors:
    print("Errors:", errors)

# Score: same bands, plus coverage
print("\n--- Score (topk) ---")
violations = 0
low_cov = 0
for query, lo, hi in SCORE_CASES:
    d = eng.decide(query).details("urgency")
    got = d["value"]
    cov = d["coverage"]
    flagged = cov < 0.28
    if flagged:
        low_cov += 1
        status = "LOW-COV (not trusted)"
    elif lo <= got <= hi:
        status = "ok"
    else:
        status = "VIOLATION"
        violations += 1
    print(f"cov={cov:.3f} band[{lo}-{hi}] -> {got:.2f}  {status:22s} | {query}")

# Flag unchanged (sanity)
flag_ok = 0
for query, expected in FLAG_CASES:
    prob = eng.decide(query).details("is_security")["probability"]
    if (prob >= 0.5) == expected:
        flag_ok += 1
print(f"\nFlag sanity: {flag_ok}/{len(FLAG_CASES)} correct")
print(f"Score: {violations} trusted-band violations, {low_cov}/{len(SCORE_CASES)} flagged low-coverage")
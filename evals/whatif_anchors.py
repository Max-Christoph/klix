"""What-if: concrete effort anchors for the task domain (queries unchanged)."""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from klix import DecisionEngine, Score  # noqa: E402
from evals.eval_domains import TASK_EFFORT_BANDS  # noqa: E402

ABSTRACT = {  # current (abstract) anchors
    "low": ["two minute quick reply", "five minute errand", "one liner fix"],
    "high": ["multi week project", "requires days of focused work", "whole team for a month"],
}
CONCRETE = {  # concrete phrasings of the same poles (NOT the test queries)
    "low": ["reply yes or no to the invitation", "send a short confirmation message", "a two minute phone call"],
    "high": ["rebuild the entire billing system from scratch", "migrate all databases over a weekend", "redesign the whole onboarding flow"],
}

for name, anchors in [("ABSTRACT (current)", ABSTRACT), ("CONCRETE (what-if)", CONCRETE)]:
    eng = DecisionEngine()
    eng.add_head(Score(name="effort", low_anchors=anchors["low"], high_anchors=anchors["high"],
                       min_val=0.0, max_val=3.0, aggregation="topk"))
    eng.compile()
    print(f"--- {name} ---")
    for query, lo, hi in TASK_EFFORT_BANDS:
        d = eng.decide(query).details("effort")
        got, cov = d["value"], d["coverage"]
        status = "LOW-COV" if cov < 0.30 else ("ok" if lo <= got <= hi else "VIOLATION")
        print(f"  cov={cov:.3f} band[{lo}-{hi}] -> {got:.2f} {status:9s} | {query}")
    print()
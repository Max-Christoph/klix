"""LLM-Guardrail / Pre-Routing scenario.

A cheap, local gatekeeper in front of an expensive LLM pipeline: route each
incoming prompt to `faq` (static answer, no LLM needed), `llm` (needs real
reasoning/generation), or `human` (escalate). Off-topic is rejected (None).

The economic point: every prompt routed to `faq` avoids an LLM call. Accuracy
here matters doubly — a missed `llm` costs latency (bad answer), a false `faq`
costs a wrong answer (worse).

Measures nearest-anchor vs linear-probe classifier on UNCHANGED anchors.
"""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from klix import DecisionEngine, Choice  # noqa: E402

OPTIONS = {
    "faq": [
        "what are your opening hours",
        "how do i reset my password",
        "where can i download the invoice",
        "how long does shipping take",
    ],
    "llm": [
        "draft a polite reply to this complaint",
        "summarize this long document for me",
        "compare these two options and recommend one",
        "write a poem about autumn",
    ],
    "human": [
        "i want to speak to a manager",
        "file a formal complaint against an employee",
        "delete all my personal data now",
    ],
}
REJECT = ["happy birthday to your team", "nice weather today", "just saying hello"]

TESTS = [
    # FAQ — answerable from a static knowledge base, no LLM needed
    ("can you tell me when you close today?", "faq"),
    ("i forgot my password, how do i reset it?", "faq"),
    ("where do i find my receipt for last month?", "faq"),
    ("how long will my order take to arrive?", "faq"),
    # LLM — needs reasoning / generation
    ("write me an email politely declining the job offer", "llm"),
    ("summarize the contract i just attached", "llm"),
    ("which of these two laptops suits video editing best?", "llm"),
    ("translate this paragraph into french for me", "llm"),
    # HUMAN — escalation
    ("get me someone in charge right now", "human"),
    ("i want to file a formal complaint", "human"),
    # Off-topic — should be rejected
    ("happy birthday to the whole team!", None),
]

OOD = ["quarterly revenue grew four percent", "please water my plants while away"]


def build(classifier: str) -> DecisionEngine:
    eng = DecisionEngine()
    eng.add_head(Choice(name="route", options=OPTIONS, reject_anchors=REJECT,
                        classifier=classifier))
    eng.compile()
    return eng


def run(classifier: str):
    eng = build(classifier)
    ok = 0
    ood_ok = 0
    errors = []
    for prompt, expected in TESTS:
        got = eng.decide(prompt).route
        if expected is None:
            # off-topic: correct = rejected (None), OR low confidence
            if got is None:
                ood_ok += 1
            else:
                errors.append((prompt, expected, got))
        elif got == expected:
            ok += 1
        else:
            errors.append((prompt, expected, got))
    n_in = len([t for t in TESTS if t[1] is not None])
    n_ood = len([t for t in TESTS if t[1] is None])
    return ok, n_in, ood_ok, n_ood, errors


if __name__ == "__main__":
    print("=" * 78)
    print("LLM-GUARDRAIL / PRE-ROUTING  (faq | llm | human | reject)")
    print("=" * 78)
    for classifier in ["nearest", "linear"]:
        ok, n_in, ood_ok, n_ood, errors = run(classifier)
        # economics: share of prompts kept away from the LLM
        eng = build(classifier)
        total = len(TESTS)
        kept_away = 0
        for prompt, _ in TESTS:
            r = eng.decide(prompt).route
            if r in ("faq", "human", None):
                kept_away += 1
        print(f"\n--- classifier={classifier:8s} ---")
        print(f"  in-domain routing : {ok}/{n_in} = {ok / n_in:.0%}")
        print(f"  off-topic rejected: {ood_ok}/{n_ood}")
        print(f"  LLM calls avoided : {kept_away}/{total} ({kept_away / total:.0%})")
        for prompt, exp, got in errors:
            print(f"    ERROR [{exp}] -> [{got}] | {prompt}")

    # Detail dump: what the gatekeeper decides per prompt (linear probe)
    print("\n--- Per-prompt decisions (linear probe) ---")
    eng = build("linear")
    for prompt, expected in TESTS:
        d = eng.decide(prompt).details("route")
        tag = f"[{expected}]" if expected else "[OOD]"
        print(f"{tag:8s} -> {str(d['value']):6s} (p={d['score']:.2f}, conf={d['confidence']:.2f}) | {prompt}")

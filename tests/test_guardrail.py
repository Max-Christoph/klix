"""LLM-Guardrail / Pre-Routing scenario tests.

The scenario: a cheap local gatekeeper in front of an expensive LLM pipeline.
It routes each prompt to `faq` (static answer), `llm` (real reasoning needed),
or `human` (escalate), and rejects off-topic as None.

These tests assert STRUCTURAL behavior (deterministic across platforms):
- the engine compiles and routes clear cases
- off-topic is rejected (None)
- the "LLM calls avoided" accounting is computable
Exact accuracy numbers are model-dependent and live in evals/eval_guardrail.py.
"""

from klix import Choice, DecisionEngine

OPTIONS = {
    "faq": [
        "what are your opening hours",
        "how do i reset my password",
        "where can i download the invoice",
    ],
    "llm": [
        "draft a polite reply to this complaint",
        "summarize this long document",
        "compare these two options and recommend one",
    ],
    "human": [
        "i want to speak to a manager",
        "file a formal complaint",
        "delete all my personal data",
    ],
}
REJECT = ["happy birthday to your team", "nice weather today", "just saying hello"]


def build(classifier: str = "nearest") -> DecisionEngine:
    eng = DecisionEngine()
    eng.add_head(Choice(name="route", options=OPTIONS, reject_anchors=REJECT,
                        classifier=classifier))
    eng.compile()
    return eng


def test_guardrail_compiles_and_routes_clear_faq():
    eng = build()
    assert eng.decide("how do i reset my password").route == "faq"


def test_guardrail_routes_escalation():
    eng = build()
    assert eng.decide("i want to speak to a manager").route == "human"


def test_guardrail_rejects_off_topic():
    eng = build()
    # Off-topic small talk should be rejected, not force-routed.
    assert eng.decide("happy birthday to the whole team").route is None


def test_guardrail_linear_probe_also_rejects():
    eng = build(classifier="linear")
    assert eng.decide("happy birthday to the whole team").route is None


def test_guardrail_llm_calls_avoided_is_computable():
    # The whole point: count how many prompts never reach the LLM.
    eng = build()
    prompts = [
        "what are your opening hours",      # faq
        "i want to speak to a manager",     # human
        "happy birthday to the team",       # reject
        "summarize this document for me",   # llm
    ]
    avoided = sum(
        1 for p in prompts if eng.decide(p).route in ("faq", "human", None)
    )
    # 3 of 4 prompts should stay away from the LLM (only the summarize one needs it)
    assert avoided == 3

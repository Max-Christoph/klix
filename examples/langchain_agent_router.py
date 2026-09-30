"""Example: Using Klix as a fast System-1 Semantic Router in LangChain and LangGraph.

Why use Klix for routing in agent pipelines?
1. Latency: Klix routes in < 15 ms on CPU vs. 800-2,000 ms for an LLM tool call.
2. Cost: 0 € API costs and 0 tokens consumed for routing decisions.
3. Multi-head power: A single 15 ms pass extracts:
   - Primary destination route (`Choice`)
   - Semantic tags (`MultiLabel`)
   - Urgency score (`Score`)
   - Security flag (`Flag`)
"""
from __future__ import annotations

import time
from klix import DecisionEngine, Choice, MultiLabel, Score, Flag
from klix.integrations.langchain import KlixRouterRunnable, create_klix_router


def setup_support_engine() -> DecisionEngine:
    """Build a multi-head engine for incoming support inquiries."""
    engine = DecisionEngine()

    # 1. Primary routing queue
    engine.add_head(
        Choice(
            name="queue",
            options={
                "it_support": ["VPN down", "laptop won't turn on", "Outlook crashing", "password reset"],
                "ot_plant":   ["robot cell stopped", "conveyor motor overheating", "PLC fault code 34"],
                "finance":    ["invoice approval needed", "cost center budget limit", "vendor payment delayed"],
            },
            reject_anchors=["lunch plans", "weekend weather", "casual greeting"],
        )
    )

    # 2. Multi-label tags
    engine.add_head(
        MultiLabel(
            name="tags",
            options={
                "hardware": ["broken screen", "cable snapped", "physical motor failure"],
                "critical": ["factory halted", "acute danger", "emergency stop triggered"],
                "network":  ["DNS resolution error", "wifi disconnected", "timeout"],
            },
            threshold=0.5,
        )
    )

    # 3. Urgency axis [0.0, 5.0]
    engine.add_head(
        Score(
            name="urgency",
            low_anchors=["whenever you have time", "routine maintenance next month"],
            high_anchors=["immediate production stop", "acute critical hazard", "emergency now"],
            min_val=0.0,
            max_val=5.0,
        )
    )

    # 4. Security flag
    engine.add_head(
        Flag(
            name="is_security",
            true_anchors=["ransomware detected", "credential leak", "suspicious root login"],
            false_anchors=["normal hardware defect", "routine update", "invoice query"],
            neutral_anchors=["general question"],
        )
    )

    engine.compile()
    return engine


def demo_langchain_runnable():
    print("=" * 70)
    print("1. LangChain Runnable: System-1 Fast Routing")
    print("=" * 70)

    engine = setup_support_engine()

    # Create a LangChain Runnable that enriches the input state with all decision heads
    router = KlixRouterRunnable(engine=engine, route_head="queue", enrich_state=True)

    test_queries = [
        {"input": "PLC-34 reports critical motor failure, assembly line halted!"},
        {"input": "Could someone please approve invoice INV-2026-402?"},
        {"input": "VPN connection drops every 5 minutes when working from home."},
        {"input": "Suspicious login from Russia detected on the admin console at 3 am."},
    ]

    for q in test_queries:
        t0 = time.perf_counter()
        state = router.invoke(q)
        dt = (time.perf_counter() - t0) * 1000

        print(f"\nQuery:    \"{q['input']}\"")
        print(f"Latency:  {dt:.1f} ms  (vs. ~1,200 ms for an LLM call)")
        print(f"Route:    --> {state['route'].upper()}")
        print(f"Tags:     {state['tags']}")
        print(f"Urgency:  {state['urgency']:.1f} / 5.0")
        print(f"Security: {state['is_security']}")


def demo_langgraph_conditional_edge():
    print("\n" + "=" * 70)
    print("2. LangGraph Conditional Edge Helper")
    print("=" * 70)

    engine = setup_support_engine()

    # create_klix_router returns a callable suitable for workflow.add_conditional_edges()
    edge_router = create_klix_router(engine, head_name="queue", input_key="messages")

    # Simulate LangGraph state containing conversation history
    class MockMessage:
        def __init__(self, content: str):
            self.content = content

    mock_state = {
        "messages": [
            MockMessage("Hi, I have a quick question."),
            MockMessage("Our welding robot 3 threw an overcurrent fault and stopped the cell."),
        ]
    }

    t0 = time.perf_counter()
    next_node = edge_router(mock_state)
    dt = (time.perf_counter() - t0) * 1000

    print(f"Last Message: \"{mock_state['messages'][-1].content}\"")
    print(f"Routed in:    {dt:.1f} ms")
    print(f"Target Node:  {next_node}")


if __name__ == "__main__":
    demo_langchain_runnable()
    demo_langgraph_conditional_edge()

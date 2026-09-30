"""Unit tests for Klix LangChain and LangGraph integration."""
from __future__ import annotations

import pytest
from klix import DecisionEngine, Choice, MultiLabel, Flag, Score
from klix.integrations.langchain import (
    KlixRouterRunnable,
    create_klix_router,
    _extract_text,
    _require_langchain,
)


@pytest.fixture
def compiled_engine():
    engine = DecisionEngine()
    engine.add_head(
        Choice(
            name="queue",
            options={
                "it_ops": ["VPN connection is down", "laptop won't boot", "server unreachable"],
                "ot_plant": ["robot cell stopped", "conveyor belt stalled", "PLC error code"],
                "finance": ["cost center over budget", "invoice approval needed", "payment delayed"],
            },
        )
    )
    engine.add_head(
        MultiLabel(
            name="tags",
            options={
                "hardware": ["laptop screen cracked", "cable snapped", "physical drive failure"],
                "critical": ["production halted", "emergency stop", "acute hazard"],
            },
            threshold=0.45,
            center=0.40,
        )
    )
    engine.compile()
    return engine


def test_klix_router_runnable_string_invoke(compiled_engine):
    router = KlixRouterRunnable(engine=compiled_engine, route_head="queue")
    route = router.invoke("VPN is down and server is unreachable")
    assert route == "it_ops"


def test_klix_router_runnable_dict_input(compiled_engine):
    router = KlixRouterRunnable(engine=compiled_engine, route_head="queue", input_key="query")
    route = router.invoke({"query": "Please approve this invoice", "ticket_id": 999})
    assert route == "finance"


def test_klix_router_runnable_output_key(compiled_engine):
    router = KlixRouterRunnable(
        engine=compiled_engine,
        route_head="queue",
        input_key="text",
        output_key="target_queue",
    )
    inp = {"text": "conveyor belt stalled in hall 3", "priority": 1}
    out = router.invoke(inp)
    assert isinstance(out, dict)
    assert out["priority"] == 1
    assert out["target_queue"] == "ot_plant"


def test_klix_router_runnable_enrich_state(compiled_engine):
    router = KlixRouterRunnable(
        engine=compiled_engine,
        route_head="queue",
        input_key="message",
        enrich_state=True,
    )
    inp = {"message": "Robot cell emergency stop, production halted!", "user": "operator1"}
    out = router.invoke(inp)
    assert isinstance(out, dict)
    assert out["user"] == "operator1"
    assert out["route"] == "ot_plant"
    assert "queue" in out
    assert "tags" in out
    assert isinstance(out["tags"], list)


def test_klix_router_runnable_batch(compiled_engine):
    router = KlixRouterRunnable(engine=compiled_engine, route_head="queue")
    queries = [
        "VPN connection dropped again",
        "robot cell stopped mid-cycle",
        "invoice payment is past due",
    ]
    routes = router.batch(queries)
    assert len(routes) == 3
    assert routes[0] == "it_ops"
    assert routes[1] == "ot_plant"
    assert routes[2] == "finance"


def test_create_klix_router_for_langgraph(compiled_engine):
    route_edge = create_klix_router(compiled_engine, head_name="queue", input_key="messages")

    # 1. State with list of strings
    state1 = {"messages": ["hello", "how can I approve this invoice?"]}
    assert route_edge(state1) == "finance"

    # 2. State with message objects having .content attribute (BaseMessage style)
    class FakeMessage:
        def __init__(self, content: str):
            self.content = content

    state2 = {
        "messages": [
            FakeMessage("hi"),
            FakeMessage("the server is unreachable and VPN is down"),
        ]
    }
    assert route_edge(state2) == "it_ops"


def test_extract_text_variations():
    assert _extract_text("simple text") == "simple text"
    assert _extract_text({"input": "dict query"}, input_key="input") == "dict query"

    class Msg:
        content = "message content"

    assert _extract_text(Msg()) == "message content"
    assert _extract_text([Msg()]) == "message content"
    assert _extract_text({"messages": [Msg()]}, input_key="messages") == "message content"


def test_require_langchain_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "langchain_core":
            raise ImportError("No module named 'langchain_core'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError, match="pip install 'klix-engine\\[langchain\\]'"):
        _require_langchain()

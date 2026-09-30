"""LangChain and LangGraph integration for Klix.

Provides ultra-fast System-1 semantic routing (< 15 ms, 0 € API cost)
before expensive LLM calls or agent tool invocations.

Usage:
    from klix import DecisionEngine, Choice, MultiLabel
    from klix.integrations.langchain import KlixRouterRunnable, create_klix_router

    # 1. As a LangChain Runnable in a chain
    router = KlixRouterRunnable(engine=engine, route_head="queue")
    chain = router | RunnableBranch(...)

    # 2. As a LangGraph conditional edge
    workflow.add_conditional_edges(
        "supervisor",
        create_klix_router(engine, head_name="queue"),
        {"it_ops": "it_node", "finance": "finance_node", None: "fallback"},
    )
"""
from __future__ import annotations

from typing import Any, Callable, Sequence


def _require_langchain():
    """Verify that langchain-core is installed."""
    try:
        import langchain_core  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "To use the Klix LangChain integration, install langchain-core:\n"
            "    pip install 'klix-engine[langchain]'\n"
            "or:\n"
            "    pip install langchain-core"
        ) from exc


def _extract_text(input_val: Any, input_key: str = "input") -> str:
    """Extract query text from string, dict, BaseMessage, or list of messages."""
    if isinstance(input_val, str):
        return input_val
    if isinstance(input_val, dict):
        val = input_val.get(input_key, "")
        if isinstance(val, str):
            return val
        if hasattr(val, "content"):
            return str(val.content)
        if isinstance(val, list) and val:
            last = val[-1]
            return getattr(last, "content", str(last))
        return str(val) if val is not None else ""
    if hasattr(input_val, "content"):
        return str(input_val.content)
    if isinstance(input_val, (list, tuple)) and input_val:
        last = input_val[-1]
        return getattr(last, "content", str(last))
    return str(input_val)


try:
    from langchain_core.runnables import Runnable, RunnableConfig
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    Runnable = object  # type: ignore
    RunnableConfig = Any  # type: ignore
    _LANGCHAIN_AVAILABLE = False


class KlixRouterRunnable(Runnable):
    """LangChain Runnable for fast semantic routing via Klix DecisionEngine.

    Parameters
    ----------
    engine : DecisionEngine
        The compiled Klix decision engine.
    route_head : str, default="queue"
        The name of the decision head to use for routing decisions.
    input_key : str, default="input"
        Key to extract text from when receiving dict input.
    output_key : str | None, default=None
        If specified, adds the route to `input_dict[output_key]` and returns the dict.
        If None and `enrich_state=False`, returns the winning route value directly.
    enrich_state : bool, default=False
        If True and input is a dict, merges all evaluated head values into the dict:
        `dict[head_name] = res.<head_name>` and `dict["route"] = res.<route_head>`.
    """

    def __init__(
        self,
        engine: Any,
        route_head: str = "queue",
        input_key: str = "input",
        output_key: str | None = None,
        enrich_state: bool = False,
    ) -> None:
        _require_langchain()
        self.engine = engine
        self.route_head = route_head
        self.input_key = input_key
        self.output_key = output_key
        self.enrich_state = enrich_state

        if not getattr(self.engine, "_compiled", False):
            self.engine.compile()

    def invoke(self, input: Any, config: RunnableConfig | None = None) -> Any:
        text = _extract_text(input, self.input_key)
        res = self.engine.decide(text)
        route_value = getattr(res, self.route_head, None)

        if self.enrich_state and isinstance(input, dict):
            out = dict(input)
            out["route"] = route_value
            for head in self.engine.heads:
                out[head.name] = getattr(res, head.name, None)
            return out

        if self.output_key and isinstance(input, dict):
            out = dict(input)
            out[self.output_key] = route_value
            return out

        return route_value

    def batch(
        self,
        inputs: Sequence[Any],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Vectorized bulk execution using engine.decide_batch() (70+ docs/s)."""
        texts = [_extract_text(inp, self.input_key) for inp in inputs]
        results = self.engine.decide_batch(texts)

        out_list = []
        for inp, res in zip(inputs, results):
            route_value = getattr(res, self.route_head, None)
            if self.enrich_state and isinstance(inp, dict):
                item = dict(inp)
                item["route"] = route_value
                for head in self.engine.heads:
                    item[head.name] = getattr(res, head.name, None)
                out_list.append(item)
            elif self.output_key and isinstance(inp, dict):
                item = dict(inp)
                item[self.output_key] = route_value
                out_list.append(item)
            else:
                out_list.append(route_value)

        return out_list


def create_klix_router(
    engine: Any,
    head_name: str = "queue",
    input_key: str = "messages",
) -> Callable[[Any], str | None]:
    """Create a conditional edge function for LangGraph graphs.

    Parameters
    ----------
    engine : DecisionEngine
        Compiled Klix DecisionEngine.
    head_name : str, default="queue"
        Decision head to extract routing label from.
    input_key : str, default="messages"
        State dictionary key containing input text or message history.

    Returns
    -------
    Callable[[Any], str | None]
        Function suitable for `workflow.add_conditional_edges(...)`.

    Example
    -------
    >>> workflow.add_conditional_edges(
    ...     "supervisor",
    ...     create_klix_router(engine, head_name="queue"),
    ...     {"it_ops": "it_node", "finance": "finance_node", None: "fallback"},
    ... )
    """
    if not getattr(engine, "_compiled", False):
        engine.compile()

    def route_edge(state: Any) -> str | None:
        text = _extract_text(state, input_key)
        res = engine.decide(text)
        return getattr(res, head_name, None)

    return route_edge


__all__ = ["KlixRouterRunnable", "create_klix_router"]

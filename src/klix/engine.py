"""The orchestration engine: binds backbone and heads together."""

import time

from klix.backbone import HybridBackbone
from klix.heads import BaseHead


class DecisionResult:
    """Result of a `decide()` call; head values accessible as attributes."""

    def __init__(self, text: str, latency_ms: float, head_data: dict):
        self.text = text
        self.latency_ms = latency_ms
        self.data = head_data

    def __getattr__(self, name: str):
        if name in self.data:
            return self.data[name]["value"]
        raise AttributeError(f"Head '{name}' does not exist in this result.")

    def details(self, name: str) -> dict:
        """Full result dict of a single head."""
        return self.data.get(name, {})

    def __repr__(self):
        items = [f"{key}={value['value']}" for key, value in self.data.items()]
        return f"<DecisionResult ({self.latency_ms:.1f}ms): {', '.join(items)}>"


class DecisionEngine:
    """Shared-backbone engine with any number of decoupled heads.

    Each `decide()` call encodes the text exactly once; all heads then evaluate
    on the same vectors.
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.backbone = HybridBackbone(model_name=model_name)
        self.heads: list[BaseHead] = []
        self._compiled = False

    def add_head(self, head: BaseHead) -> "DecisionEngine":
        """Registers a head (fluent, chainable)."""
        self.heads.append(head)
        self._compiled = False
        return self

    def compile(self) -> None:
        """Collects all reference texts from all heads and initializes the indexes."""
        all_texts = []
        for head in self.heads:
            all_texts.extend(head.get_reference_texts())

        if not all_texts:
            raise ValueError("No reference texts found: register heads via add_head() first.")

        self.backbone.build_vocabulary(all_texts)

        for head in self.heads:
            head.fit(self.backbone)

        self._compiled = True

    def decide(self, text: str) -> DecisionResult:
        """Encodes the text once and evaluates all heads on the vectors."""
        if not self.heads:
            raise ValueError("No heads registered: call add_head() first.")
        if not self._compiled:
            self.compile()

        start = time.perf_counter()

        # 1. One-time vectorization (~10-12 ms).
        encoded = self.backbone.encode(text)

        # 2. Evaluation of all heads (< 1 ms total).
        results: dict[str, dict] = {}
        for head in self.heads:
            results[head.name] = head.evaluate(encoded)

        elapsed_ms = (time.perf_counter() - start) * 1000
        return DecisionResult(text=text, latency_ms=elapsed_ms, head_data=results)
"""The orchestration engine: binds backbone and heads together."""

import time

from klix.backbone import HybridBackbone
from klix.heads import BaseHead


class DecisionResult:
    """Result of a `decide()` call; head values accessible as attributes."""

    def __init__(self, text: str, latency_ms: float, head_data: dict, engine=None):
        self.text = text
        self.latency_ms = latency_ms
        self.data = head_data
        self._engine = engine

    def __getattr__(self, name: str):
        if name in self.data:
            return self.data[name]["value"]
        raise AttributeError(f"Head '{name}' does not exist in this result.")

    def details(self, name: str) -> dict:
        """Full result dict of a single head."""
        return self.data.get(name, {})

    def explain(self, name: str) -> dict:
        """Human-readable explanation of one head's decision.

        Delegates to the head's ``explain(text, details)`` method. The Choice
        head explains which keywords and which anchor drove the decision (exact
        token attribution via the sparse channel); Score and Flag explain the
        pole similarities. Raises ValueError for unknown heads.
        """
        if name not in self.data:
            raise ValueError(f"Head '{name}' does not exist in this result.")
        if self._engine is None:
            raise ValueError("explain() is unavailable on this result (no engine reference).")
        head = next((h for h in self._engine.heads if h.name == name), None)
        if head is None or not hasattr(head, "explain_decision"):
            return {"note": f"head '{name}' does not support explanations"}
        return head.explain_decision(self.text, self.data.get(name, {}), backbone=self._engine.backbone)

    def __repr__(self):
        items = [f"{key}={value['value']}" for key, value in self.data.items()]
        return f"<DecisionResult ({self.latency_ms:.1f}ms): {', '.join(items)}>"


class DecisionEngine:
    """Shared-backbone engine with any number of decoupled heads.

    Each `decide()` call encodes the text exactly once; all heads then evaluate
    on the same vectors.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        stop_words: list[str] | None = None,
    ):
        self.backbone = HybridBackbone(model_name=model_name)
        self.stop_words = stop_words
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

        self.backbone.build_vocabulary(all_texts, stop_words=self.stop_words)

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
        return DecisionResult(text=text, latency_ms=elapsed_ms, head_data=results, engine=self)

    def decide_batch(self, texts: list[str]) -> list[DecisionResult]:
        """Processes many texts in one embedding pass (bulk mode).

        The dense encoder runs once over the whole list (fastembed batches the
        ONNX forward pass internally), so per-text overhead drops sharply for
        large volumes. Head evaluation then loops per text but stays in the
        sub-millisecond regime. Results are returned in input order; latency_ms
        per item reflects its share of the batch time.

        For pipelines that already hold embeddings, `encode_batch` +
        head.evaluate() can be used directly to avoid re-encoding.
        """
        if not self.heads:
            raise ValueError("No heads registered: call add_head() first.")
        if not self._compiled:
            self.compile()
        if not texts:
            return []

        start = time.perf_counter()

        # 1. ONE embedding pass over the whole batch.
        encoded_list = self.backbone.encode_batch(texts)

        # 2. Head evaluation per encoded input (heads are microsecond-fast;
        #    a vectorized multi-head matmul would complicate the BaseHead API
        #    for no measurable gain at this scale).
        results: list[DecisionResult] = []
        for text, encoded in zip(texts, encoded_list):
            head_data: dict[str, dict] = {}
            for head in self.heads:
                head_data[head.name] = head.evaluate(encoded)
            results.append(DecisionResult(text=text, latency_ms=0.0, head_data=head_data, engine=self))

        elapsed_ms = (time.perf_counter() - start) * 1000
        per_item = elapsed_ms / max(1, len(texts))
        for r in results:
            r.latency_ms = per_item
        return results

    def calibrate(self, head_name: str, samples: list, metric: str = "f1") -> dict:
        """Learns decision parameters for one head from labeled validation samples.

        - Flag: samples = [(text, bool)], calibrates ``threshold`` via the chosen
          metric (f1|precision|recall|accuracy).
        - Score: samples = [(text, float_target)], grid-searches ``sharpness``
          and fits an affine remap (documented in Score.calibrate).
        - Choice: samples = [(text, label_or_None)], calibrates ``reject_threshold``.

        Returns the head's report dict; raises ValueError for unknown heads or
        heads without a calibrate method (e.g. custom heads).
        """
        head = next((h for h in self.heads if h.name == head_name), None)
        if head is None:
            raise ValueError(f"Head '{head_name}' is not registered.")
        if not self._compiled:
            self.compile()
        if not hasattr(head, "calibrate"):
            raise ValueError(f"Head '{head_name}' ({type(head).__name__}) has no calibrate().")
        return head.calibrate(self.backbone, samples, metric=metric)
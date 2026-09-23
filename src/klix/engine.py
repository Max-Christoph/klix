"""The orchestration engine: binds backbone and heads together."""

import time

from klix.backbone import HybridBackbone
from klix.heads import BaseHead, Choice, Flag, Score


def _klix_version() -> str:
    try:
        from klix import __version__
        return __version__
    except Exception:
        return "unknown"


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
        max_chars: int | None = 2000,
        smart_truncate: bool = True,
    ) -> None:
        self.backbone = HybridBackbone(
            model_name=model_name,
            max_chars=max_chars,
            smart_truncate=smart_truncate,
        )
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
        self._schema_hash = self.schema_hash()

    def schema_hash(self) -> str:
        """Stable SHA-256 over the full schema configuration (E.3).

        Covers every input that determines decisions: heads and their
        parameters, anchors, counterexamples, rules, calibrated thresholds and
        stop words. Log it alongside every decision batch so historical
        results stay reproducible — there is no versioned trained model to
        point at, the schema hash IS the model version.
        """
        import hashlib
        import json as _json

        def _head_state(h):
            state = {
                "type": type(h).__name__,
                "name": h.name,
            }
            if isinstance(h, Choice):
                state.update({
                    # Anchor lists are semantically unordered sets for the
                    # nearest path; sort them so the hash is canonical over
                    # insertion order (verified by test_insertion_order_
                    # invariant — without sorting, the hash differs).
                    "options": {k: sorted(v) for k, v in h.options.items()},
                    "keyword_boost": h.keyword_boost,
                    "reject_anchors": sorted(h.reject_anchors),
                    "label_aggregation": h.label_aggregation,
                    "label_topk": h.label_topk,
                    "keyword_boost_mode": h.keyword_boost_mode,
                    "reject_threshold": h.reject_threshold,
                    "classifier": h.classifier,
                    "classifier_C": h.classifier_C,
                    "sparse_metric": h.sparse_metric,
                    "bm25_k1": h.bm25_k1,
                    "bm25_b": h.bm25_b,
                    "counterexamples": {k: sorted(v) for k, v in h.counterexamples.items()},
                    "rules": sorted(rule.describe() for _regex, rule in h._compiled_rules),
                })
            elif isinstance(h, Score):
                state.update({
                    "low_anchors": sorted(h.low_anchors),
                    "high_anchors": sorted(h.high_anchors),
                    "min_val": h.min_val,
                    "max_val": h.max_val,
                    "sharpness": h.sharpness,
                    "aggregation": h.aggregation,
                    "topk": h.topk,
                    "min_coverage": h.min_coverage,
                    "fallback_value": h.fallback_value,
                    "soft_coverage": h.soft_coverage,
                    "calib_a": h._calib_a,
                    "calib_b": h._calib_b,
                })
            elif isinstance(h, Flag):
                state.update({
                    "true_anchors": sorted(h.true_anchors),
                    "false_anchors": sorted(h.false_anchors),
                    "neutral_anchors": sorted(h.neutral_anchors),
                    "threshold": h.threshold,
                    "temp": h.temp,
                    "aggregation": h.aggregation,
                    "topk": h.topk,
                })
            else:
                # Custom head subclass: serialize __dict__ best-effort (sorted
                # keys, str values) so the hash still covers its configuration.
                state["state"] = {
                    k: str(v) for k, v in sorted(vars(h).items())
                    if not k.startswith("_")
                }
            return state

        payload = {
            "version": _klix_version(),
            "stop_words": self.stop_words,
            "model_name": getattr(self.backbone.embed_model, "model_name", None),
            "max_chars": self.backbone.max_chars,
            "smart_truncate": self.backbone.smart_truncate,
            "heads": [_head_state(h) for h in self.heads],
        }
        blob = _json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def decide(self, text: str) -> DecisionResult:
        """Encodes the text once and evaluates all heads on the vectors."""
        if not self.heads:
            raise ValueError("No heads registered: call add_head() first.")
        if not self._compiled:
            self.compile()

        start = time.perf_counter()

        # 1. One-time vectorization (~10-12 ms).
        encoded = self.backbone.encode(text)

        # 2. Evaluation of all heads (< 1 ms total), with head gating:
        #    a head whose suppress_when condition matches an earlier head's
        #    result is skipped (value=None, suppressed_by marker).
        results: dict[str, dict] = {}
        for head in self.heads:
            if head.is_suppressed(results):
                results[head.name] = {"value": None, "suppressed_by": sorted(head.suppress_when)}
            else:
                results[head.name] = head.evaluate(encoded)

        elapsed_ms = (time.perf_counter() - start) * 1000
        return DecisionResult(text=text, latency_ms=elapsed_ms, head_data=results, engine=self)

    def decide_batch(self, texts: list[str]) -> list[DecisionResult]:
        """Processes many texts in one embedding pass (bulk mode).

        The dense encoder runs once over the whole list (fastembed batches the
        ONNX forward pass internally). Honest measurements (Windows/CPU,
        MiniLM, 2 heads) — the ONNX forward cost itself scales ~linearly with
        token count, so only the fixed call overhead amortizes:

            n=5:    ~1.5x faster than serial decide() per item
            n=25:   ~3.1x
            n=500:  ~2.5-3.2x  (~3 ms/item)

        Use decide() for single interactive items; decide_batch() pays off
        from roughly 25+ texts. Results are returned in input order; latency_ms
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
                if head.is_suppressed(head_data):
                    head_data[head.name] = {"value": None, "suppressed_by": sorted(head.suppress_when)}
                else:
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

    def validate_anchors(self) -> list[dict]:
        """Read-only anchor-quality report for all Choice heads.

        Detects per head:
        - overlapping classes (pairwise centroid cosine >= 0.75), with the
          shared confuser terms and per-side sharpening hints
        - misplaced anchors (closer to a foreign centroid than to their own)
        - structural issues (classes with 1 anchor, duplicate anchors)

        The report NEVER mutates anchors; applying suggestions stays with the
        developer. Returns a list of findings dicts (see klix.validate).
        """
        if not self._compiled:
            self.compile()
        from klix.validate import validate_choice_head

        report: list[dict] = []
        for head in self.heads:
            if hasattr(head, "options"):  # Choice-like heads only
                report.extend(validate_choice_head(head))
        return report

    def validate_anchors_report(self) -> str:
        """Formatted human-readable version of validate_anchors()."""
        findings = self.validate_anchors()
        if not findings:
            return "All anchors look well separated. No findings."
        lines = []
        for f in findings:
            if f["kind"] == "overlap":
                marker = "!!" if f["severity"] == "high" else " ?"
                lines.append(
                    f"{marker} [{f['head']}] '{f['a']}' <-> '{f['b']}': "
                    f"centroid cosine {f['centroid_cos']} ({f['severity']})"
                )
                if f["shared_terms"]:
                    lines.append(f"      shared confusers: {', '.join(f['shared_terms'][:5])}")
                if f["a_exclusive"]:
                    lines.append(f"      sharpen '{f['a']}' with: {', '.join(f['a_exclusive'][:3])}")
                if f["b_exclusive"]:
                    lines.append(f"      sharpen '{f['b']}' with: {', '.join(f['b_exclusive'][:3])}")
                for m in f["misplaced"][:3]:
                    lines.append(f"      misplaced: {m['text']!r} ({m['from']}) sits closer to '{m['closer_to']}'")
                for s in f.get("suggestions", []):
                    if s:
                        lines.append(f"      fix: {s}")
            else:
                lines.append(f" ?  [{f['head']}] {f['message']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialization (v0.8.0): warm start for serverless/container restarts.
    # compile() costs model loading + anchor embedding; save()/load() skips
    # both by persisting the fitted state (head config + precomputed matrices
    # + fitted TF-IDF). The embedding MODEL itself is not persisted (it is
    # cached by fastembed on disk anyway); load() re-instantiates it from the
    # model_name. A loaded engine behaves identically to a freshly compiled
    # one — verified by the round-trip tests.
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persists the compiled engine state to `path` (joblib binary).

        Saves head configurations, precomputed anchor matrices, fitted TF-IDF
        vectorizers, calibrated thresholds and compiled rules. The embedding
        model itself is NOT saved (fastembed caches the download); load()
        re-creates it from the stored model_name.
        """
        import joblib

        if not self._compiled:
            self.compile()
        state = {
            "version": self.backbone and getattr(self.backbone, "model_name", None),
            "model_name": getattr(self.backbone.embed_model, "model_name", None),
            "stop_words": self.stop_words,
            "max_chars": self.backbone.max_chars,
            "smart_truncate": self.backbone.smart_truncate,
            "tfidf_vec": self.backbone.tfidf_vec,
            "heads": self.heads,
        }
        joblib.dump(state, path, compress=3)

    @classmethod
    def load(cls, path: str) -> "DecisionEngine":
        """Restores an engine saved with save(). Heads are restored in their
        fitted state; the first decide() does not need compile()."""
        import joblib

        state = joblib.load(path)
        eng = cls(
            stop_words=state["stop_words"],
            max_chars=state["max_chars"],
            smart_truncate=state["smart_truncate"],
        )
        # NOTE: the embedding model re-instantiates here (~1-2 s cold start
        # for ONNX session init, but no anchor re-embedding).
        eng.backbone.tfidf_vec = state["tfidf_vec"]
        eng.backbone.is_indexed = state["tfidf_vec"] is not None
        eng.heads = list(state["heads"])
        eng._compiled = True
        return eng
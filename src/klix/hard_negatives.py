"""Hard-negative mining: collect low-confidence live decisions and turn them
into counterexamples for the most-confused labels.

Workflow (D.3):
1. Observe live traffic with `HardNegativeStore.observe(result)` — call it
   after every decide(), or use `engine.attach_hard_negative_store(store)`.
2. `store.mine(head_name, min_margin)` extracts the cases where the runner-up
   was close to the winner (small margin = confusion risk): these are the
   hard negatives, the pairs of classes the schema cannot separate yet.
3. Feed them back with `Choice.add_counterexamples(label, texts)` — label
   being the label the text was WRONGLY pulled toward — and `compile()`
   again. Counterexamples act as label-specific reject poles: text close to
   a counterexample of label X loses similarity credit for X specifically.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .heads import Choice


class HardNegativeStore:
    """Collects low-confidence decisions from live traffic.

    The store keeps the raw decision dicts (one JSON object per line on disk)
    and derives mining candidates from the score margins. It never blocks
    inference: observe() is a dict append, microseconds.
    """

    def __init__(self, path: str | None = None):
        self.path = path
        self._cases: list[dict] = []
        if path and Path(path).exists():
            with open(path, encoding="utf-8") as fh:
                self._cases = [json.loads(line) for line in fh if line.strip()]

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------
    def observe(self, result) -> None:
        """Records one DecisionResult (all heads) into the store."""
        for head_name, data in result.data.items():
            if not isinstance(data, dict) or "value" not in data:
                continue
            self._cases.append({
                "ts": time.time(),
                "text": result.text,
                "head": head_name,
                "value": data.get("value"),
                "score": data.get("score"),
                "confidence": data.get("confidence"),
                "scores": data.get("scores") or {},
            })
        if self.path:
            self._flush()

    def _flush(self) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(self._cases[-1], ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._cases)

    # ------------------------------------------------------------------
    # Mining
    # ------------------------------------------------------------------
    def mine(self, head_name: str, min_margin: float = 0.05, min_score: float = 0.0) -> list[dict]:
        """Extracts hard negatives for one head.

        A case is a hard negative when the winner beat the runner-up by less
        than ``min_margin`` — the model was barely sure, and a single anchor
        word can flip such decisions. Returns dicts with text, picked label,
        runner-up label and the margin, sorted by margin (most confused first).
        """
        out = []
        for case in self._cases:
            if case["head"] != head_name or case["value"] is None:
                continue
            scores = case.get("scores") or {}
            if len(scores) < 2:
                continue
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            (best_label, best_score), (runner_label, runner_score) = ranked[0], ranked[1]
            margin = best_score - runner_score
            if margin <= min_margin and best_score >= min_score:
                out.append({
                    "text": case["text"],
                    "picked": best_label,
                    "runner_up": runner_label,
                    "margin": round(margin, 4),
                })
        out.sort(key=lambda c: c["margin"])
        return out


def attach_counterexamples(head: Choice, mined: list[dict], max_per_label: int = 20) -> dict[str, list[str]]:
    """Routes mined cases into the head as counterexamples.

    For each mined case the text was *pulled* toward ``picked`` while
    ``runner_up`` was nearly as close. The text is added as a counterexample
    for ``picked``: it describes what that label is NOT (or at least what the
    schema confused with runner_up). Returns the counterexample map applied.

    The caller is responsible for labelling (a human decides whether the
    picked label was right); unreviewed auto-application is deliberately not
    supported — automated negative training without labels poisons schemas.
    """
    applied: dict[str, list[str]] = {}
    for case in mined:
        label = case["picked"]
        bucket = applied.setdefault(label, [])
        if case["text"] not in bucket and len(bucket) < max_per_label:
            bucket.append(case["text"])
    for label, texts in applied.items():
        head.add_counterexamples(label, texts)
    return applied
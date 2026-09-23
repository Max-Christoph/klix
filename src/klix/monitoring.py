"""Drift monitoring (E.2): watch live coverage/confidence distributions.

A production schema can silently drift: new traffic mixes in wording the
anchors never saw, and every decision still *looks* confident. The monitor
tracks the streaming distribution of the per-head trust signals (Score
coverage, Flag margin, Choice confidence) and raises a callback when the
share of low-trust decisions exceeds a threshold — the cheap, statistical
early-warning that new failure types exist before users report them.
"""

from __future__ import annotations

import time
from collections import deque


class DriftMonitor:
    """Streaming monitor over decision trust signals.

    Usage:
        mon = DriftMonitor(warn_fn=print)
        res = engine.decide(text)
        mon.observe(res)          # after every decide (or batch)

    The callback fires at most once per ``cooldown_s`` seconds when the share
    of decisions whose trust signal fell below ``low_cut`` exceeds
    ``low_share`` over the sliding window.
    """

    #: per-head-type trust signal extractors (head result dict -> float or None)
    SIGNALS = {
        "Score": lambda d: d.get("coverage"),
        "Flag": lambda d: d.get("margin"),
        "Choice": lambda d: d.get("confidence"),
    }

    def __init__(
        self,
        window: int = 500,
        low: float = 0.30,
        low_share: float = 0.35,
        warn_fn=None,
        cooldown_s: float = 300.0,
    ):
        self.window = window
        self.low = low
        self.low_share = low_share
        self.warn_fn = warn_fn
        self.cooldown_s = cooldown_s
        self._buf: deque[dict] = deque(maxlen=window)
        self._last_warn = 0.0

    def observe(self, result) -> None:
        """Records one DecisionResult's per-head trust signals."""
        entry: dict[str, float] = {}
        for head_name, data in result.data.items():
            if not isinstance(data, dict):
                continue
            sig = self._extract(head_name, data)
            if sig is not None:
                entry[head_name] = sig
        if entry:
            self._cases = getattr(self, "_cases", deque(maxlen=self.window))
            self._cases.append(entry)
            self._check()

    def _extract(self, head_name: str, data: dict) -> float | None:
        # The head type is not stored on the dict; infer from the fields.
        if "coverage" in data and "raw_diff" in data:
            return data["coverage"]          # Score
        if "margin" in data:
            return data["margin"]            # Flag
        if "confidence" in data and data.get("value") is not None:
            return data["confidence"]        # Choice
        return None

    def stats(self) -> dict:
        """Current per-head mean trust signal + low-trust share."""
        cases = getattr(self, "_cases", [])
        out: dict[str, dict] = {}
        for head in (set().union(*[set(c) for c in cases]) if cases else set()):
            vals = [c[head] for c in cases if head in c]
            if not vals:
                continue
            out[head] = {
                "n": len(vals),
                "mean": round(sum(vals) / len(vals), 4),
                "low_share": round(sum(v < self.low for v in vals) / len(vals), 4),
            }
        return out

    def _check(self) -> None:
        if self.warn_fn is None:
            return
        stats = self.stats()
        drifted = {h: s for h, s in stats.items() if s["low_share"] >= self.low_share}
        if not drifted:
            return
        now = time.time()
        if now - self._last_warn < self.cooldown_s:
            return
        self._last_warn = now
        try:
            self.warn_fn({
                "message": "drift warning: share of low-trust decisions above threshold",
                "low_threshold": self.low,
                "low_share_required": self.low_share,
                "heads": drifted,
            })
        except Exception:
            pass  # monitoring must never break inference
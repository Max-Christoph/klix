"""Declarative rule overlay for Choice heads: hard keyword/regex signals that
override or boost the semantic decision.

Rules are the strict counterpart to the semantic channel. They always apply
after the classifier (nearest or linear probe) has produced scores, and a
``force`` rule wins over everything — including the reject pole — because it is
the user's most explicit signal.

Examples:
    Rule(label="ot_plant", any_of=["notfall", "stillstand"], mode="force")
    Rule(label="security", pattern=r"\\b10\\.\\d+\\.\\d+\\.\\d+\\b", mode="boost", weight=2.0)
"""

import re
from dataclasses import dataclass

VALID_MODES = ("force", "boost")


@dataclass
class Rule:
    """A hard rule evaluated against the raw input text.

    Attributes:
        label:    The Choice option label this rule applies to. Must exist in
                  the head's ``options`` (validated at compile time).
        any_of:   Keywords (case-insensitive, word-boundary matched). Either
                  ``any_of`` or ``pattern`` is required.
        pattern:  A raw regular expression (alternative to ``any_of``).
        mode:     "force" -> on match, the label wins immediately.
                  "boost" -> on match, ``weight`` is added to the label's score.
        weight:   Added to the label score for "boost". NOTE: in the
                  ``classifier="linear"`` path scores are probabilities in
                  [0, 1], so a weight >= 1.0 effectively acts like "force".
        name:     Optional human-readable id, surfaced in ``matched_rules``.
    """

    label: str
    any_of: list[str] | None = None
    pattern: str | None = None
    mode: str = "boost"
    weight: float = 2.0
    name: str = ""

    def __post_init__(self):
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {self.mode!r}")
        if not self.any_of and not self.pattern:
            raise ValueError("Rule needs either any_of keywords or a pattern")
        if self.mode == "boost" and self.weight <= 0:
            raise ValueError("boost rules need weight > 0")

    def compile(self) -> re.Pattern:
        """Compiles this rule into a case-insensitive regex over the raw text."""
        if self.pattern:
            return re.compile(self.pattern, re.IGNORECASE)
        words = "|".join(re.escape(w) for w in (self.any_of or []))
        return re.compile(rf"(?i)\b(?:{words})\b")

    def describe(self) -> str:
        """Human-readable matcher description used in details/matched_rules."""
        if self.name:
            return self.name
        if self.pattern:
            return f"pattern:{self.pattern}"
        return f"keywords:{','.join(self.any_of or [])}"


def compile_rules(rules: list) -> list[tuple[re.Pattern, Rule]]:
    """Compiles a list of Rule objects into (regex, rule) pairs."""
    return [(rule.compile(), rule) for rule in rules]
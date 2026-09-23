"""Klix: decoupled decision heads (Choice, Score, Flag) on a shared semantic backbone."""

from klix.engine import DecisionEngine, DecisionResult
from klix.heads import BaseHead, Choice, Flag, Score
from klix.rules import Rule

__version__ = "0.7.2"

__all__ = [
    "DecisionEngine",
    "DecisionResult",
    "BaseHead",
    "Choice",
    "Score",
    "Flag",
    "Rule",
]
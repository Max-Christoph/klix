"""Klix: decoupled decision heads (Choice, Score, Flag) on a shared semantic backbone."""

from klix.engine import DecisionEngine, DecisionResult
from klix.hard_negatives import HardNegativeStore, attach_counterexamples
from klix.heads import BaseHead, Choice, Flag, Score
from klix.monitoring import DriftMonitor
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
    "HardNegativeStore",
    "attach_counterexamples",
    "DriftMonitor",
]
"""Klix: Entkoppelte Entscheidungs-Köpfe (Choice, Score, Flag) auf einem geteilten semantischen Backbone."""

from klix.engine import DecisionEngine, DecisionResult
from klix.heads import BaseHead, Choice, Flag, Score

__version__ = "0.1.2"

__all__ = [
    "DecisionEngine",
    "DecisionResult",
    "BaseHead",
    "Choice",
    "Score",
    "Flag",
]
"""Klix: decoupled decision heads (Choice, Score, Flag) on a shared semantic backbone."""

from klix.engine import DecisionEngine, DecisionResult
from klix.glossary import Glossary, load_glossary, resolve_glossary
from klix.glossaries import merge_all
from klix.glossaries import default as default_glossary
from klix.glossaries import empty as empty_glossary
from klix.glossaries import manufacturing as manufacturing_glossary
from klix.glossaries import workflow as workflow_glossary
from klix.hard_negatives import HardNegativeStore, attach_counterexamples
from klix.heads import BaseHead, Choice, Flag, Score
from klix.monitoring import DriftMonitor
from klix.rules import Rule

__version__ = "0.9.0"

__all__ = [
    "DecisionEngine",
    "DecisionResult",
    "BaseHead",
    "Choice",
    "Score",
    "Flag",
    "Rule",
    "Glossary",
    "load_glossary",
    "resolve_glossary",
    # Domain packs (klix.glossaries)
    "empty_glossary",
    "manufacturing_glossary",
    "workflow_glossary",
    "default_glossary",
    "merge_all",
    "HardNegativeStore",
    "attach_counterexamples",
    "DriftMonitor",
]

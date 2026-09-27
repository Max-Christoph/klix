"""Klix: decoupled decision heads (Choice, Score, Flag) on a shared semantic backbone."""

from klix.engine import DecisionEngine, DecisionResult
from klix.glossary import Glossary, load_glossary, resolve_glossary
from klix.glossaries import merge_all
from klix.glossaries import broad as broad_glossary
from klix.glossaries import curated as curated_glossary
from klix.glossaries import curated_everyday as curated_everyday_glossary
from klix.glossaries import curated_it as curated_it_glossary
from klix.glossaries import curated_manufacturing as curated_manufacturing_glossary
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
    "curated_glossary",
    "curated_manufacturing_glossary",
    "curated_it_glossary",
    "curated_everyday_glossary",
    "broad_glossary",
    "manufacturing_glossary",
    "workflow_glossary",
    "merge_all",
    "HardNegativeStore",
    "attach_counterexamples",
    "DriftMonitor",
]

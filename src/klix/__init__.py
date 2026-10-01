"""Klix: decoupled decision heads (Choice, Score, Flag) on a shared semantic backbone."""

from klix import langid
from klix.engine import DecisionEngine, DecisionResult
from klix.glossary import (SCHEMA_VERSION, SUPPORTED_LANGUAGES, Glossary,
                           GlossaryConflict, GlossaryRegistry, load_glossary,
                           resolve_glossary)
from klix.glossaries import basic_manufacturing as basic_manufacturing_glossary
from klix.glossaries import curated_where as curated_where_glossary
from klix.glossaries import language_packs
from klix.glossaries import merge_all
from klix.glossaries import broad as broad_glossary
from klix.glossaries import curated as curated_glossary
from klix.glossaries import curated_everyday as curated_everyday_glossary
from klix.glossaries import curated_it as curated_it_glossary
from klix.glossaries import curated_manufacturing as curated_manufacturing_glossary
from klix.glossaries import empty as empty_glossary
from klix.glossaries import manufacturing as manufacturing_glossary
from klix.glossaries import multilingual as multilingual_glossary
from klix.glossaries import workflow as workflow_glossary
from klix.hard_negatives import HardNegativeStore, attach_counterexamples
from klix.heads import BaseHead, Choice, Flag, MultiLabel, Score
from klix.monitoring import DriftMonitor
from klix.rules import Rule

__version__ = "0.12.3"

__all__ = [
    "DecisionEngine",
    "DecisionResult",
    "BaseHead",
    "Choice",
    "Score",
    "Flag",
    "MultiLabel",
    "Rule",
    "Glossary",
    "GlossaryConflict",
    "GlossaryRegistry",
    "SCHEMA_VERSION",
    "SUPPORTED_LANGUAGES",
    "load_glossary",
    "resolve_glossary",
    # Language identification (klix.langid)
    "langid",
    # Domain packs (klix.glossaries)
    "empty_glossary",
    "curated_glossary",
    "curated_manufacturing_glossary",
    "curated_it_glossary",
    "curated_everyday_glossary",
    "curated_where_glossary",
    "broad_glossary",
    "basic_manufacturing_glossary",
    "manufacturing_glossary",
    "multilingual_glossary",
    "workflow_glossary",
    "language_packs",
    "merge_all",
    "HardNegativeStore",
    "attach_counterexamples",
    "DriftMonitor",
]

"""Shared feature extraction: a text enters exactly once, transformed dense + sparse."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from fastembed import TextEmbedding
from sklearn.feature_extraction.text import TfidfVectorizer

# Compact bilingual (EN + DE) stopword list — deliberately small so domain terms
# keep their signal. `stop_words=None` (default) uses this list, `stop_words=[]`
# disables stopword filtering entirely, and a custom list replaces it (e.g. for
# a third language). Extendable via DecisionEngine(stop_words=...).
_DEFAULT_STOPWORDS = [
    # German
    "die", "der", "das", "ein", "eine", "einer", "eines", "einem", "einen",
    "im", "in", "ist", "und", "für", "von", "mit", "an", "auf", "nach", "zu",
    "nicht", "mehr", "wird", "wie", "was", "hier", "dort",
    # English
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "and", "for",
    "of", "with", "to", "in", "on", "at", "do", "does", "did", "this", "that",
    "it", "i", "you", "we", "they", "not", "have", "has", "had", "will", "can",
]

# Backward-compatible alias (older code imported _DEFAULT_GERMAN_STOPWORDS).
_DEFAULT_GERMAN_STOPWORDS = _DEFAULT_STOPWORDS


@dataclass
class EncodedInput:
    """Result of the one-time feature extraction for a single input text."""

    text: str
    dense_vec: np.ndarray  # normalized dense vector (MiniLM, 384 dim)
    sparse_vec: Any  # TF-IDF sparse matrix (1 x V) or None before compile()


class HybridBackbone:
    """Encapsulates the dense (FastEmbed) and sparse (TF-IDF) representation of a text.

    Instantiated exactly once by the engine. Each `decide()` call encodes the text
    once; all heads then operate on the resulting `EncodedInput`.
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.embed_model = TextEmbedding(model_name=model_name)
        self.tfidf_vec: TfidfVectorizer | None = None
        self.is_indexed = False

    def build_vocabulary(
        self,
        all_texts: list[str],
        stop_words: list[str] | None = None,
    ) -> None:
        """Builds the sparse index over all reference texts registered in the heads."""
        if stop_words is None:
            stop_words = _DEFAULT_STOPWORDS
        self.stop_words = stop_words
        self.tfidf_vec = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b[\w-]+\b",
            lowercase=True,
            stop_words=stop_words,
        )
        self.tfidf_vec.fit(all_texts)
        self.is_indexed = True

    def encode(self, text: str) -> EncodedInput:
        """Produces both vectors in a single pass."""
        vec = np.array(list(self.embed_model.embed([text]))[0])
        norm = float(np.linalg.norm(vec))
        dense_norm = vec / (norm if norm > 0 else 1.0)

        sparse = self.tfidf_vec.transform([text]) if self.is_indexed else None
        return EncodedInput(text=text, dense_vec=dense_norm, sparse_vec=sparse)
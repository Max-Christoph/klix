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
#
# Lesson (the "Joghurt-Fehler"): grammatical fillers like "den"/"hat" gave the
# sparse channel high weights on off-domain queries and overrode semantics.
# Articles, pronouns, auxiliaries and prepositions carry no routing signal and
# are therefore filtered by default.
_DEFAULT_STOPWORDS = [
    # German: articles, pronouns, auxiliaries, prepositions, common adverbs
    "die", "der", "das", "den", "dem", "des", "ein", "eine", "einer", "eines",
    "einem", "einen", "einer", "ich", "du", "er", "sie", "es", "wir", "ihr",
    "mich", "dir", "uns", "mir", "sich", "im", "in", "ist", "bin", "bist",
    "sind", "war", "waren", "und", "oder", "für", "von", "mit", "an", "auf",
    "nach", "zu", "zum", "zur", "bei", "aus", "über", "unter", "vor", "hinter",
    "nicht", "kein", "keine", "mehr", "wird", "werden", "wurde", "wurden",
    "wie", "was", "wer", "hier", "dort", "hat", "hatte", "haben", "hatte",
    "seit", "schon", "noch", "nur", "auch", "wieder", "um", "dann", "als",
    # German politeness/filler particles ("bitte" dominated TF-IDF confuser lists)
    "bitte", "danke", "dank", "vielen", "mal", "gerne", "vielleicht", "hallo",
    # English: articles, pronouns, auxiliaries, prepositions
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "for", "of", "with", "to", "in", "on", "at", "by", "from",
    "do", "does", "did", "this", "that", "these", "those", "it", "i", "you",
    "he", "she", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "our", "their", "its", "not", "have", "has", "had", "will", "can", "could",
    "would", "should", "may", "might", "must", "shall", "again", "still",
    "just", "also", "only", "than", "then", "there", "here", "what", "which",
    "who", "whom", "how", "when", "where", "why", "all", "each", "every",
    "some", "any", "no", "nor", "not", "so", "too", "very",
    # English politeness/filler words
    "please", "thanks", "thank", "hello", "hi", "kindly",
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

    Input truncation (v0.8.0): real-world inputs are often long emails with
    quoted history and disclaimers. MiniLM truncates at 512 tokens anyway —
    everything past the limit is silently ignored while still costing CPU
    time. `max_chars` (default 2000) caps the input before encoding;
    `smart_truncate=True` first tries the first paragraph/section break
    (support tickets almost always state their intent there) instead of a
    blunt character cut mid-sentence.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        max_chars: int | None = 2000,
        smart_truncate: bool = True,
    ):
        self.embed_model = TextEmbedding(model_name=model_name)
        self.tfidf_vec: TfidfVectorizer | None = None
        self.is_indexed = False
        self.max_chars = max_chars
        self.smart_truncate = smart_truncate

    def _truncate(self, text: str) -> str:
        """Caps input length. smart mode prefers a paragraph/section break."""
        if self.max_chars is None or len(text) <= self.max_chars:
            return text
        if self.smart_truncate:
            # prefer cutting at a blank line, newline, then sentence end
            window = text[: self.max_chars]
            for sep in ["\n\n", "\n", ". ", " "]:
                cut = window.rfind(sep)
                if cut >= self.max_chars // 2:
                    return text[: cut + len(sep)].rstrip()
        return text[: self.max_chars]

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
        """Produces both vectors in a single pass (input is truncated first)."""
        text = self._truncate(text)
        vec = np.array(list(self.embed_model.embed([text]))[0])
        norm = float(np.linalg.norm(vec))
        dense_norm = vec / (norm if norm > 0 else 1.0)

        sparse = self.tfidf_vec.transform([text]) if self.is_indexed else None
        return EncodedInput(text=text, dense_vec=dense_norm, sparse_vec=sparse)

    def encode_batch(self, texts: list[str]) -> list[EncodedInput]:
        """Encodes many texts in ONE dense pass (bulk mode, truncated first).

        fastembed batches the ONNX forward internally, so per-text overhead
        drops sharply versus calling encode() in a loop. The sparse transform is
        a single sklearn call over the whole list.
        """
        if not texts:
            return []
        texts = [self._truncate(t) for t in texts]
        dense = np.array(list(self.embed_model.embed(texts)))
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        dense = dense / np.where(norms == 0, 1.0, norms)
        sparse = self.tfidf_vec.transform(texts) if self.is_indexed else None
        return [
            EncodedInput(text=text, dense_vec=dense[i], sparse_vec=(sparse[i] if sparse is not None else None))
            for i, text in enumerate(texts)
        ]
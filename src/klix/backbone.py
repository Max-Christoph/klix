"""Geteilte Feature-Extraktion: ein Text geht genau einmal dense + sparse transformiert rein."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from fastembed import TextEmbedding
from sklearn.feature_extraction.text import TfidfVectorizer

# Kompakte deutsche Stoppwortliste (bewusst klein, damit Fachbegriffe ihre
# Information behalten; erweiterbar via build_vocabulary(..., stop_words=...)).
_DEFAULT_GERMAN_STOPWORDS = [
    "die", "der", "das", "ein", "eine", "einer", "eines", "einem", "einen",
    "im", "in", "ist", "und", "für", "von", "mit", "an", "auf", "nach", "zu",
    "nicht", "mehr", "wird", "wie", "was", "hier", "dort",
]


@dataclass
class EncodedInput:
    """Ergebnis der einmaligen Feature-Extraktion für einen Eingabetext."""

    text: str
    dense_vec: np.ndarray  # normalisierter Dense-Vektor (MiniLM, 384 dim)
    sparse_vec: Any  # TF-IDF Sparse-Matrix (1 x V) oder None vor compile()


class HybridBackbone:
    """Kapselt Dense- (FastEmbed) und Sparse- (TF-IDF) Repräsentation eines Texts.

    Wird von der Engine genau einmal instanziiert. Der Text wird pro `decide()`-Aufruf
    genau einmal encoded; alle Köpfe arbeiten anschließend auf `EncodedInput`.
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
        """Baut den Sparse-Index über alle in den Köpfen hinterlegten Referenztexte auf."""
        if stop_words is None:
            stop_words = _DEFAULT_GERMAN_STOPWORDS
        self.tfidf_vec = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b[\w-]+\b",
            lowercase=True,
            stop_words=stop_words,
        )
        self.tfidf_vec.fit(all_texts)
        self.is_indexed = True

    def encode(self, text: str) -> EncodedInput:
        """Erzeugt beide Vektoren in einem Rutsch."""
        vec = np.array(list(self.embed_model.embed([text]))[0])
        norm = float(np.linalg.norm(vec))
        dense_norm = vec / (norm if norm > 0 else 1.0)

        sparse = self.tfidf_vec.transform([text]) if self.is_indexed else None
        return EncodedInput(text=text, dense_vec=dense_norm, sparse_vec=sparse)
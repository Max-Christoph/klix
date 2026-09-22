"""Die Entscheidungsköpfe: Choice (Routing), Score (Achse), Flag (Boolesch).

Alle Köpfe arbeiten ausschließlich auf den vorberechneten Vektoren aus
`EncodedInput` und sind dadurch voneinander entkoppelt.
"""

from abc import ABC, abstractmethod

import numpy as np

from klix.backbone import EncodedInput, HybridBackbone


class BaseHead(ABC):
    """Basisklasse für alle Köpfe.

    Ein Kopf wird dreiphasig verwendet:
    1. `get_reference_texts()` — sammelt Referenztexte für den TF-IDF-Index.
    2. `fit(backbone)` — Vorberechnung aller Referenzvektoren (einmalig).
    3. `evaluate(encoded)` — Bewertung pro Abfrage, < 0.1 ms Ziel.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_reference_texts(self) -> list[str]:
        """Gibt alle Texte zurück, die der Backbone für den TF-IDF-Index kennen muss."""

    @abstractmethod
    def fit(self, backbone: HybridBackbone) -> None:
        """Vorberechnung von Referenzvektoren."""

    @abstractmethod
    def evaluate(self, encoded: EncodedInput) -> dict:
        """Berechnet das Ergebnis basierend auf den vorberechneten Vektoren."""


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Zeilenweise auf Einheitslänge normieren (0-Vektoren bleiben 0)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


class Choice(BaseHead):
    """Klassifikation / Routing über Max-Similarity + Keyword-Boost.

    Options-Label mit den meisten ähnlichen Beispielsätzen gewinnt. Die Sparse-
    Ähnlichkeit (exakte Worttreffer, z. B. Asset-IDs wie `plc-34`) wird mit
    `keyword_boost` auf die Dense-Ähnlichkeit addiert.
    """

    def __init__(self, name: str, options: dict[str, list[str]], keyword_boost: float = 0.5):
        super().__init__(name)
        self.options = options
        self.keyword_boost = keyword_boost
        self.flat_texts: list[str] = []
        self.label_map: list[str] = []
        self.dense_matrix: np.ndarray | None = None
        self.sparse_matrix = None

    def get_reference_texts(self) -> list[str]:
        return [text for examples in self.options.values() for text in examples]

    def fit(self, backbone: HybridBackbone) -> None:
        self.flat_texts = []
        self.label_map = []
        for label, examples in self.options.items():
            for example in examples:
                self.flat_texts.append(example)
                self.label_map.append(label)

        vecs = np.array(list(backbone.embed_model.embed(self.flat_texts)))
        self.dense_matrix = _normalize_rows(vecs)
        self.sparse_matrix = backbone.tfidf_vec.transform(self.flat_texts)

    def evaluate(self, encoded: EncodedInput) -> dict:
        dense_sims = self.dense_matrix @ encoded.dense_vec

        # Direkter Sparse-Dot statt sklearn cosine_similarity: TfidfVectorizer
        # normiert beide Vektoren L2 (default norm="l2"), der Dot nicht-negativer
        # Einheitsvektoren IST die Cosine-Aehnlichkeit — aber ~5x schneller
        # (kein sklearn-Call-Overhead pro Abfrage).
        sparse_sims = np.asarray((encoded.sparse_vec @ self.sparse_matrix.T).todense())[0]

        hybrid_sims = dense_sims + self.keyword_boost * sparse_sims

        # Pro Label die beste Beispiel-Ähnlichkeit behalten.
        category_scores: dict[str, float] = {}
        for idx, sim in enumerate(hybrid_sims):
            label = self.label_map[idx]
            if label not in category_scores or sim > category_scores[label]:
                category_scores[label] = float(sim)

        best_label = max(category_scores, key=category_scores.get)
        best_score = category_scores[best_label]

        # Margin zum Runner-Up als Konfidenzkalibrierung.
        sorted_scores = sorted(category_scores.values(), reverse=True)
        runner_up = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        confidence = float(np.clip((best_score - runner_up) / (best_score + 1e-5) * 1.5, 0.0, 1.0))

        return {
            "value": best_label,
            "score": best_score,
            "confidence": confidence,
            "scores": category_scores,
        }


class Score(BaseHead):
    """Kontinuierliche Projektion auf eine semantische Achse.

    Der Text wird gegen Low- und High-Anker similarity-gemessen; die Differenz
    geht durch eine Sigmoid-Scherfunktion und wird auf [min_val, max_val] gemappt.
    """

    def __init__(
        self,
        name: str,
        low_anchors: list[str],
        high_anchors: list[str],
        min_val: float = 0.0,
        max_val: float = 3.0,
        sharpness: float = 8.0,
    ):
        super().__init__(name)
        self.low_anchors = low_anchors
        self.high_anchors = high_anchors
        self.min_val = min_val
        self.max_val = max_val
        self.sharpness = sharpness
        self.low_matrix: np.ndarray | None = None
        self.high_matrix: np.ndarray | None = None

    def get_reference_texts(self) -> list[str]:
        return self.low_anchors + self.high_anchors

    def fit(self, backbone: HybridBackbone) -> None:
        low_v = np.array(list(backbone.embed_model.embed(self.low_anchors)))
        self.low_matrix = _normalize_rows(low_v)

        high_v = np.array(list(backbone.embed_model.embed(self.high_anchors)))
        self.high_matrix = _normalize_rows(high_v)

    def evaluate(self, encoded: EncodedInput) -> dict:
        # Max-Similarity zu beiden Polen.
        s_low = float(np.max(self.low_matrix @ encoded.dense_vec))
        s_high = float(np.max(self.high_matrix @ encoded.dense_vec))

        # Sigmoid-basierte Skalierung der Differenz.
        diff = s_high - s_low
        ratio = 1.0 / (1.0 + np.exp(-diff * self.sharpness))
        calculated_score = self.min_val + ratio * (self.max_val - self.min_val)

        return {
            "value": round(float(calculated_score), 2),
            "raw_diff": diff,
        }


class Flag(BaseHead):
    """Boolesche Entscheidung mit kalibrierter Wahrscheinlichkeit.

    Softmax über die Similarities zu True- und False-Ankern; Temperatur steuert
    die Schärfe der Entscheidung.

    Problem ohne drittes Pol: Bei Out-of-Domain-Texten sind beide Similarities
    niedrig und nahe beieinander -> Wahrscheinlichkeit ~0.5 und Rauschen kippt
    die Entscheidung. Mit `neutral_anchors` wird ein 3-Klassen-Softmax genutzt;
    der Kopf liefert dann `value=None` (statt True/False), wenn "neutral"
    gewinnt. Standardmäßig deaktiviert (klassisches 2-Klassen-Verhalten).
    """

    def __init__(
        self,
        name: str,
        true_anchors: list[str],
        false_anchors: list[str],
        threshold: float = 0.5,
        temp: float = 0.12,
        neutral_anchors: list[str] | None = None,
    ):
        super().__init__(name)
        self.true_anchors = true_anchors
        self.false_anchors = false_anchors
        self.threshold = threshold
        self.temp = temp
        self.neutral_anchors = neutral_anchors or []
        self.true_matrix: np.ndarray | None = None
        self.false_matrix: np.ndarray | None = None
        self.neutral_matrix: np.ndarray | None = None

    def get_reference_texts(self) -> list[str]:
        return self.true_anchors + self.false_anchors + self.neutral_anchors

    def fit(self, backbone: HybridBackbone) -> None:
        t_v = np.array(list(backbone.embed_model.embed(self.true_anchors)))
        self.true_matrix = _normalize_rows(t_v)

        f_v = np.array(list(backbone.embed_model.embed(self.false_anchors)))
        self.false_matrix = _normalize_rows(f_v)

        if self.neutral_anchors:
            n_v = np.array(list(backbone.embed_model.embed(self.neutral_anchors)))
            self.neutral_matrix = _normalize_rows(n_v)
        else:
            self.neutral_matrix = None

    def evaluate(self, encoded: EncodedInput) -> dict:
        s_true = float(np.max(self.true_matrix @ encoded.dense_vec))
        s_false = float(np.max(self.false_matrix @ encoded.dense_vec))

        # Softmax über zwei (oder drei) Klassen mit Temperatur-Skalierung.
        logits = np.array([s_true, s_false], dtype=float)
        if self.neutral_matrix is not None:
            s_neutral = float(np.max(self.neutral_matrix @ encoded.dense_vec))
            logits = np.array([s_true, s_false, s_neutral], dtype=float)

        scaled = logits / self.temp
        scaled -= scaled.max()  # numerisch stabiler Softmax
        exp = np.exp(scaled)
        probs = exp / exp.sum()

        if self.neutral_matrix is not None:
            prob_true, prob_false, prob_neutral = (float(p) for p in probs)
            if prob_neutral > max(prob_true, prob_false):
                return {
                    "value": None,
                    "probability": prob_true,
                    "probabilities": {"true": prob_true, "false": prob_false, "neutral": prob_neutral},
                }
            value = bool(prob_true >= self.threshold)
            return {
                "value": value,
                "probability": prob_true,
                "probabilities": {"true": prob_true, "false": prob_false, "neutral": prob_neutral},
            }

        prob = float(probs[0])
        return {
            "value": bool(prob >= self.threshold),
            "probability": prob,
        }
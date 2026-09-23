"""The decision heads: Choice (routing), Score (axis), Flag (boolean).

All heads operate exclusively on the precomputed vectors from `EncodedInput`,
which makes them fully decoupled from each other.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from klix.backbone import _DEFAULT_GERMAN_STOPWORDS, EncodedInput, HybridBackbone


def _make_local_vectorizer(stop_words: list[str] | None) -> TfidfVectorizer:
    """Creates a TF-IDF vectorizer scoped to a single head.

    Per-head vectorizers keep heads truly decoupled: IDF weights are computed
    from the head's own reference texts only, so adding heads or options never
    changes another head's keyword scores.
    """
    return TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b[\w-]+\b",
        lowercase=True,
        stop_words=stop_words if stop_words is not None else _DEFAULT_GERMAN_STOPWORDS,
    )


class BaseHead(ABC):
    """Base class for all heads.

    A head is used in three phases:
    1. `get_reference_texts()` — collects reference texts for the TF-IDF index.
    2. `fit(backbone)` — precomputes all reference vectors (once).
    3. `evaluate(encoded)` — evaluates per query, target < 0.1 ms.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_reference_texts(self) -> list[str]:
        """Returns all texts the backbone must know for the TF-IDF index."""

    @abstractmethod
    def fit(self, backbone: HybridBackbone) -> None:
        """Precomputes reference vectors."""

    @abstractmethod
    def evaluate(self, encoded: EncodedInput) -> dict:
        """Computes the result based on the precomputed vectors."""


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization (zero vectors stay zero)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


class Choice(BaseHead):
    """Classification / routing via max-similarity + keyword boost.

    The option label with the most similar example sentences wins. Sparse
    similarity (exact word hits, e.g. asset IDs like `plc-34`) is added to the
    dense similarity, weighted by `keyword_boost`.

    Optional `reject_anchors` describe what the head should NOT classify
    (small talk, off-topic requests). When the best reject anchor matches more
    strongly than any option, the head returns `value=None` — a "don't know"
    signal instead of a forced guess. This catches clearly out-of-domain texts;
    borderline cases still pass through (check `confidence`).
    """

    def __init__(
        self,
        name: str,
        options: dict[str, list[str]],
        keyword_boost: float = 0.5,
        reject_anchors: list[str] | None = None,
    ):
        super().__init__(name)
        self.options = options
        self.keyword_boost = keyword_boost
        self.reject_anchors = reject_anchors or []
        self.reject_matrix: np.ndarray | None = None
        self.flat_texts: list[str] = []
        self.label_map: list[str] = []
        self.dense_matrix: np.ndarray | None = None
        self.sparse_matrix = None
        self.tfidf = None  # per-head vectorizer (see _make_local_vectorizer)

    def get_reference_texts(self) -> list[str]:
        texts = [text for examples in self.options.values() for text in examples]
        return texts + self.reject_anchors

    def fit(self, backbone: HybridBackbone) -> None:
        self.flat_texts = []
        self.label_map = []
        for label, examples in self.options.items():
            for example in examples:
                self.flat_texts.append(example)
                self.label_map.append(label)

        vecs = np.array(list(backbone.embed_model.embed(self.flat_texts)))
        self.dense_matrix = _normalize_rows(vecs)

        # Own vocabulary + IDF weights, scoped to this head's reference texts.
        self.tfidf = _make_local_vectorizer(getattr(backbone, "stop_words", None))
        self.sparse_matrix = self.tfidf.fit_transform(self.flat_texts)
        # Cache vocabulary/IDF for the hand-rolled query vectorizer in evaluate().
        self._vocab = self.tfidf.vocabulary_
        self._idf = self.tfidf.idf_

        if self.reject_anchors:
            r_v = np.array(list(backbone.embed_model.embed(self.reject_anchors)))
            self.reject_matrix = _normalize_rows(r_v)
        else:
            self.reject_matrix = None

    def _sparse_query_vec(self, text: str):
        """TF-IDF vector for `text` using this head's cached vocabulary/IDF.

        Mirrors sklearn's TfidfVectorizer (word ngrams=(1,1), sublinear_tf=False,
        L2 norm) exactly, but skips the per-query sklearn call overhead (~0.5 ms).
        """
        import re

        tokens = re.findall(r"(?u)\b[\w-]+\b", text.lower())
        counts: dict[int, float] = {}
        for tok in tokens:
            col = self._vocab.get(tok)
            if col is not None:  # vocabulary_ already excludes stop words
                counts[col] = counts.get(col, 0.0) + 1.0
        if not counts:
            return None
        # tf * idf, then L2 normalize
        vec = {col: tf * self._idf[col] for col, tf in counts.items()}
        norm = float(np.sqrt(sum(v * v for v in vec.values())))
        if norm > 0:
            vec = {col: v / norm for col, v in vec.items()}
        return vec

    def evaluate(self, encoded: EncodedInput) -> dict:
        dense_sims = self.dense_matrix @ encoded.dense_vec

        # Hand-rolled TF-IDF query vector (same math as sklearn, ~40x faster:
        # no per-query sklearn transform call). Dotted against the L2-normalized
        # reference matrix, the result IS cosine similarity.
        query_vec = self._sparse_query_vec(encoded.text)
        if query_vec:
            cols = np.fromiter(query_vec.keys(), dtype=np.int64)
            vals = np.fromiter(query_vec.values(), dtype=float)
            row = csr_matrix(
                (vals, (np.zeros(len(cols), dtype=int), cols)),
                shape=(1, len(self._vocab)),
            )
            sparse_sims = np.asarray((row @ self.sparse_matrix.T).todense()).ravel()
        else:
            sparse_sims = np.zeros(self.sparse_matrix.shape[0])

        hybrid_sims = dense_sims + self.keyword_boost * sparse_sims

        # Keep the best example similarity per label.
        category_scores: dict[str, float] = {}
        for idx, sim in enumerate(hybrid_sims):
            label = self.label_map[idx]
            if label not in category_scores or sim > category_scores[label]:
                category_scores[label] = float(sim)

        best_label = max(category_scores, key=category_scores.get)
        best_score = category_scores[best_label]

        # Optional reject pole: wins against every option -> "don't know".
        reject_sim = 0.0
        if self.reject_matrix is not None:
            reject_sim = float(np.max(self.reject_matrix @ encoded.dense_vec))
            if reject_sim > best_score:
                return {
                    "value": None,
                    "score": best_score,
                    "confidence": 0.0,
                    "scores": category_scores,
                    "reject_score": reject_sim,
                }

        # Margin to the runner-up as confidence calibration.
        sorted_scores = sorted(category_scores.values(), reverse=True)
        runner_up = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        confidence = float(np.clip((best_score - runner_up) / (best_score + 1e-5) * 1.5, 0.0, 1.0))

        return {
            "value": best_label,
            "score": best_score,
            "confidence": confidence,
            "scores": category_scores,
            "reject_score": reject_sim,
        }


class Score(BaseHead):
    """Continuous projection onto a semantic axis.

    The text is similarity-measured against low and high anchors; the difference
    passes through a sigmoid sharpening function and is mapped to
    [min_val, max_val].

    `aggregation` controls how anchor similarities are pooled:
    - `"max"` (default, backward compatible): single best anchor decides.
    - `"topk"`: mean of the best `topk` anchors per pole — robust against a
      single noisy anchor, recommended when you have 3+ anchors per pole.

    Every result additionally carries a `coverage` value (the pooled similarity
    to the better pole). Low coverage means the text did not resemble either
    pole — the score is then mostly noise and should not be trusted.
    """

    def __init__(
        self,
        name: str,
        low_anchors: list[str],
        high_anchors: list[str],
        min_val: float = 0.0,
        max_val: float = 3.0,
        sharpness: float = 8.0,
        aggregation: str = "max",
        topk: int = 2,
    ):
        super().__init__(name)
        self.low_anchors = low_anchors
        self.high_anchors = high_anchors
        self.min_val = min_val
        self.max_val = max_val
        self.sharpness = sharpness
        self.aggregation = aggregation
        self.topk = topk
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
        low_sims = self.low_matrix @ encoded.dense_vec
        high_sims = self.high_matrix @ encoded.dense_vec

        if self.aggregation == "topk":
            k = max(1, min(self.topk, len(low_sims), len(high_sims)))
            s_low = float(np.mean(np.sort(low_sims)[-k:]))
            s_high = float(np.mean(np.sort(high_sims)[-k:]))
        else:
            s_low = float(np.max(low_sims))
            s_high = float(np.max(high_sims))

        # Sigmoid-based scaling of the difference.
        diff = s_high - s_low
        ratio = 1.0 / (1.0 + np.exp(-diff * self.sharpness))
        calculated_score = self.min_val + ratio * (self.max_val - self.min_val)

        return {
            "value": round(float(calculated_score), 2),
            "raw_diff": diff,
            "coverage": float(max(s_low, s_high)),
        }


class Flag(BaseHead):
    """Boolean decision with calibrated probability.

    Softmax over the similarities to true and false anchors; temperature controls
    the sharpness of the decision.

    Problem without a third pole: for out-of-domain texts both similarities are
    low and close together -> probability ~0.5 and noise flips the decision.
    With `neutral_anchors` a 3-class softmax is used; the head then returns
    `value=None` (instead of True/False) whenever "neutral" wins. Disabled by
    default (classic 2-class behavior).
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

        # Softmax over two (or three) classes with temperature scaling.
        logits = np.array([s_true, s_false], dtype=float)
        if self.neutral_matrix is not None:
            s_neutral = float(np.max(self.neutral_matrix @ encoded.dense_vec))
            logits = np.array([s_true, s_false, s_neutral], dtype=float)

        scaled = logits / self.temp
        scaled -= scaled.max()  # numerically stable softmax
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
"""The decision heads: Choice (routing), Score (axis), Flag (boolean).

All heads operate exclusively on the precomputed vectors from `EncodedInput`,
which makes them fully decoupled from each other.
"""

from abc import ABC, abstractmethod

import re

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from klix.backbone import _DEFAULT_STOPWORDS, EncodedInput, HybridBackbone

# Sentinel class label for the optional reject class when classifier="linear".
_REJECT_LABEL = "__klix_reject__"


# Cheap German-signal words (umlaut-free forms included). Used by _detect_lang
# for classifier="auto" and cross-lingual mixup — a heuristic, not a detector.
_GERMAN_SIGNAL_WORDS = [
    "der", "die", "das", "und", "ist", "nicht", "eine", "ein", "mit", "für",
    "auf", "nach", "von", "im", "in", "mein", "meine", "wurde", "wird", "haben",
    "fehlt", "kaputt", "staendig", "bricht", "startet", "konto", "rechnung",
    "bestellung", "heizung", "gehaltsabrechnung", "urlaub", "kreditkarte",
    "erstattung", "verschluesselt", "loesegeld", "unbekannte", "einloggt",
    "gutschrift", "doppelt", "abgebucht", "belastet", "monat", "kueche", "tropft",
    "wlan", "verbindet", "bildschirm", "schwarz", "zerbrochen", "tuerknauf",
    "elternzeit", "abrechnung", "stunden", "postfach", "dateien",
]


def _detect_lang(text: str) -> str:
    """Cheap language heuristic for anchor texts.

    Returns "de" if the text contains German-specific characters (umlauts/sharp-s)
    or German signal words, else "en". Used only for `classifier="auto"` and
    cross-lingual augmentation — not a general-purpose language detector. For
    better detection, prefer supplying `translate_fn` or keeping anchors in one
    language.
    """
    low = text.lower()
    for ch in low:
        if ch in "äöüß":
            return "de"
    words = re.findall(r"(?u)\b[\w-]+\b", low)
    for w in words:
        if w in _GERMAN_SIGNAL_WORDS:
            return "de"
    return "en"


def _cross_lingual_mixup(X: np.ndarray, y: np.ndarray, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Augments each class with midpoints between its anchors in DIFFERENT languages.

    When a class has anchors in both EN and DE, the embedding-space midpoint
    between an English and a German anchor is a point that sits between the two
    languages — teaching the downstream probe that the class is one cloud, not
    two language-split clouds. This is the model-free part of cross-lingual
    augmentation (real translation still requires `translate_fn`).
    """
    parts_x: list[np.ndarray] = [X]
    parts_y: list[np.ndarray] = [y]
    groups: dict[str, list[int]] = {}
    for i, lab in enumerate(y):
        groups.setdefault(lab, []).append(i)
    for lab, idxs in groups.items():
        langs = [_detect_lang(texts[i]) for i in idxs]
        if len(set(langs)) <= 1:  # skip: class is single-language
            continue
        for a, lang_a in zip(idxs, langs):
            for b, lang_b in zip(idxs, langs):
                if lang_a != lang_b:
                    mid = X[a] + X[b]
                    n = float(np.linalg.norm(mid))
                    mid = mid / n if n > 0 else mid
                    parts_x.append(mid.reshape(1, -1))
                    parts_y.append(np.array([lab]))
    return np.vstack(parts_x), np.concatenate(parts_y)


def _augment_embeddings(
    X: np.ndarray,
    y: np.ndarray,
    mixup: bool = True,
    noise_std: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Expands a few-shot anchor set with same-class embedding interpolation.

    - mixup: pairwise midpoints of anchors within the same class (densifies each
      class cloud without any external text source).
    - noise: a small amount of normalized Gaussian noise (teaches a downstream
      linear probe a smoother decision boundary).

    Runs at compile time only; inference cost is unchanged.
    """
    rng = np.random.RandomState(seed)
    parts_x: list[np.ndarray] = [X]
    parts_y: list[np.ndarray] = [y]
    if mixup:
        groups: dict = {}
        for i, lab in enumerate(y):
            groups.setdefault(lab, []).append(i)
        for lab, idxs in groups.items():
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    mid = X[idxs[a]] + X[idxs[b]]
                    n = float(np.linalg.norm(mid))
                    mid = mid / n if n > 0 else mid
                    parts_x.append(mid.reshape(1, -1))
                    parts_y.append(np.array([lab]))
    if noise_std > 0:
        noisy = _normalize_rows(X + rng.normal(0.0, noise_std, X.shape))
        parts_x.append(noisy)
        parts_y.append(y)
    return np.vstack(parts_x), np.concatenate(parts_y)


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
        stop_words=stop_words if stop_words is not None else _DEFAULT_STOPWORDS,
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

    Additional knobs:
    - `label_aggregation="topk"` pools the best `label_topk` anchors per label
      (robust against a single lucky anchor; recommended with 4+ anchors/label).
    - `keyword_boost_mode="coverage"` damps the keyword channel by the fraction
      of query tokens this head's vocabulary recognizes (off-language/off-domain
      queries stop over-boosting generic words).
    - `reject_threshold` (a hybrid-score floor) forces `value=None` below it.
    """

    def __init__(
        self,
        name: str,
        options: dict[str, list[str]],
        keyword_boost: float = 0.5,
        reject_anchors: list[str] | None = None,
        label_aggregation: str = "max",
        label_topk: int = 2,
        keyword_boost_mode: str = "fixed",
        reject_threshold: float | None = None,
        classifier: str = "nearest",
        classifier_C: float = 10.0,
        translate_fn=None,
    ):
        super().__init__(name)
        self.options = options
        self.keyword_boost = keyword_boost
        self.reject_anchors = reject_anchors or []
        self.reject_matrix: np.ndarray | None = None
        self.reject_sparse_matrix = None
        self.label_aggregation = label_aggregation
        self.label_topk = label_topk
        self.keyword_boost_mode = keyword_boost_mode
        self.reject_threshold = reject_threshold
        self.classifier = classifier
        self.classifier_C = classifier_C
        self.translate_fn = translate_fn
        self._probe = None  # LogisticRegression probe when classifier="linear"
        self._probe_labels: list[str] = []
        self.flat_texts: list[str] = []
        self.label_map: list[str] = []
        self._label_rows: dict[str, list[int]] = {}
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
        # Fit on options AND reject anchors so the reject pole's distinctive
        # terms are part of the vocabulary (otherwise its sparse score is 0).
        self.tfidf = _make_local_vectorizer(getattr(backbone, "stop_words", None))
        self.tfidf.fit(self.flat_texts + self.reject_anchors)
        self.sparse_matrix = self.tfidf.transform(self.flat_texts)
        # Cache vocabulary/IDF for the hand-rolled query vectorizer in evaluate().
        self._vocab = self.tfidf.vocabulary_
        self._idf = self.tfidf.idf_
        # Group row indices by label (for label pooling).
        self._label_rows = {}
        for row, label in enumerate(self.label_map):
            self._label_rows.setdefault(label, []).append(row)

        if self.reject_anchors:
            r_v = np.array(list(backbone.embed_model.embed(self.reject_anchors)))
            self.reject_matrix = _normalize_rows(r_v)
            # Sparse vectors for the reject anchors too, so the reject pole can
            # be scored on the same hybrid scale as the options (see evaluate).
            self.reject_sparse_matrix = self.tfidf.transform(self.reject_anchors)
        else:
            self.reject_matrix = None
            self.reject_sparse_matrix = None

        # --- Resolve classifier="auto" -> "nearest" or "linear" -----------------
        # If anchors are mixed-language (EN+DE), the linear probe overfits to the
        # language with more anchors; nearest-anchor is more robust there. For
        # single-language schemas, the probe is the accuracy winner.
        effective_classifier = self.classifier
        if effective_classifier == "auto":
            langs = {_detect_lang(t) for t in self.flat_texts}
            effective_classifier = "nearest" if len(langs) > 1 else "linear"

        # --- Train a linear probe on the (augmented) anchor embeddings --------
        # A learned decision boundary separates overlapping class clouds far
        # better than nearest-anchor distance; with frozen embeddings this is the
        # standard few-shot approach (linear probing). Trained here, at compile
        # time, on the anchors only — inference is a single 384-dim matmul.
        if effective_classifier == "linear":
            X = self.dense_matrix  # already L2-normalized rows
            y = np.array(self.label_map)
            texts = self.flat_texts
            if self.reject_matrix is not None:
                # Reject anchors become their own class so the probe learns a
                # boundary against off-domain text (returns None when it wins).
                X = np.vstack([X, self.reject_matrix])
                y = np.concatenate([y, np.array([_REJECT_LABEL] * self.reject_matrix.shape[0])])
                texts = texts + [""] * self.reject_matrix.shape[0]

            # Cross-lingual augmentation: bridge language-split classes.
            if self.translate_fn is not None:
                # User supplied a translator: mirror each anchor to the other
                # language and add it to the same class.
                new_texts, new_labels = [], []
                for text, lab in zip(texts, y):
                    other = _detect_lang(text)
                    target = "de" if other == "en" else "en"
                    try:
                        translated = self.translate_fn(text, target)
                        if translated:
                            new_texts.append(translated)
                            new_labels.append(lab)
                    except Exception:
                        pass  # translation is best-effort; never break compile
                if new_texts:
                    tvecs = np.array(list(backbone.embed_model.embed(new_texts)))
                    X = np.vstack([X, _normalize_rows(tvecs)])
                    y = np.concatenate([y, np.array(new_labels)])
                    texts = texts + new_texts  # keep texts aligned with X/y

            # Cross-lingual mixup + standard augmentation.
            X_cl, y_cl = _cross_lingual_mixup(X, y, texts)
            X_aug, y_aug = _augment_embeddings(X_cl, y_cl)
            self._probe = LogisticRegression(
                C=self.classifier_C,
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",
            )
            self._probe.fit(X_aug, y_aug)
            self._probe_labels = list(self._probe.classes_)
            self._effective_classifier = "linear"
        else:
            self._probe = None
            self._probe_labels = []
            self._effective_classifier = "nearest"

    def _sparse_query_vec(self, text: str) -> tuple[dict[int, float] | None, float]:
        """TF-IDF vector for `text` using this head's cached vocabulary/IDF.

        Mirrors sklearn's TfidfVectorizer (word ngrams=(1,1), sublinear_tf=False,
        L2 norm) exactly, but skips the per-query sklearn call overhead (~0.5 ms).

        Returns ``(vec, coverage)`` where ``coverage`` is the fraction of the
        query's tokens this head's vocabulary recognizes (0.0 for an empty or
        fully out-of-vocabulary query, 1.0 for a fully in-vocabulary query).
        """
        tokens = re.findall(r"(?u)\b[\w-]+\b", text.lower())
        if not tokens:
            return None, 0.0
        counts: dict[int, float] = {}
        matched = 0
        for tok in tokens:
            col = self._vocab.get(tok)
            if col is not None:  # vocabulary_ already excludes stop words
                counts[col] = counts.get(col, 0.0) + 1.0
                matched += 1
        coverage = matched / len(tokens)
        if not counts:
            return None, coverage
        # tf * idf, then L2 normalize
        vec = {col: tf * self._idf[col] for col, tf in counts.items()}
        norm = float(np.sqrt(sum(v * v for v in vec.values())))
        if norm > 0:
            vec = {col: v / norm for col, v in vec.items()}
        return vec, coverage

    def evaluate(self, encoded: EncodedInput) -> dict:
        # --- Linear-probe path (classifier="linear") -------------------------
        if self._probe is not None:
            # Single matmul + softmax over the probe; dense only, microseconds.
            probs = self._probe.predict_proba(encoded.dense_vec.reshape(1, -1))[0]
            label_probs = dict(zip(self._probe_labels, probs))
            best_label = max(label_probs, key=label_probs.get)
            best_prob = label_probs[best_label]
            if best_label == _REJECT_LABEL:
                # Off-domain: reject class won -> "don't know".
                return {
                    "value": None,
                    "score": float(best_prob),
                    "confidence": 0.0,
                    "scores": {k: float(v) for k, v in label_probs.items() if k != _REJECT_LABEL},
                    "reject_score": float(best_prob),
                }
            # Confidence from probability margin (top class minus runner-up).
            option_probs = [v for k, v in label_probs.items() if k != _REJECT_LABEL]
            sorted_probs = sorted(option_probs, reverse=True)
            runner_up = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
            confidence = float(np.clip(best_prob - runner_up, 0.0, 1.0))
            return {
                "value": best_label,
                "score": float(best_prob),
                "confidence": confidence,
                "scores": {k: float(v) for k, v in label_probs.items() if k != _REJECT_LABEL},
                "reject_score": float(label_probs.get(_REJECT_LABEL, 0.0)),
            }

        dense_sims = self.dense_matrix @ encoded.dense_vec

        # Hand-rolled TF-IDF query vector (same math as sklearn, ~40x faster:
        # no per-query sklearn transform call). Dotted against the L2-normalized
        # reference matrix, the result IS cosine similarity.
        query_vec, query_coverage = self._sparse_query_vec(encoded.text)
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

        # --- Keyword channel -----------------------------------------------
        if self.keyword_boost_mode == "coverage":
            # Damp the keyword channel by the fraction of query tokens the
            # vocabulary recognizes: off-language/off-domain queries stop
            # over-boosting generic words that happen to overlap.
            boost = self.keyword_boost * query_coverage
        else:  # "fixed" (default, backward compatible)
            boost = self.keyword_boost
        hybrid_sims = dense_sims + boost * sparse_sims

        # --- Pool per label -------------------------------------------------
        if self.label_aggregation == "topk":
            # k is chosen per label, so a label with fewer anchors does not
            # silently force all labels down to a max()-style pool.
            category_scores = {
                label: float(np.mean(np.sort(hybrid_sims[rows])[-min(self.label_topk, len(rows)):]))
                for label, rows in self._label_rows.items()
            }
        else:  # "max" (default, backward compatible)
            category_scores = {}
            for idx, sim in enumerate(hybrid_sims):
                label = self.label_map[idx]
                if label not in category_scores or sim > category_scores[label]:
                    category_scores[label] = float(sim)

        best_label = max(category_scores, key=category_scores.get)
        best_score = category_scores[best_label]

        # --- Reject pole on the SAME scale as the best score -----------------
        # The reject anchors participate in the hybrid score too (dense + the
        # same keyword channel), so the comparison is apples-to-apples.
        reject_sim = 0.0
        if self.reject_matrix is not None:
            reject_dense = float(np.max(self.reject_matrix @ encoded.dense_vec))
            # keyword score of the query against reject anchors
            if query_vec:
                reject_sparse = np.asarray(
                    (row @ self.reject_sparse_matrix.T).todense()
                ).ravel().max() if self.reject_sparse_matrix is not None else 0.0
            else:
                reject_sparse = 0.0
            reject_sim = reject_dense + boost * reject_sparse
            if reject_sim > best_score:
                return {
                    "value": None,
                    "score": best_score,
                    "confidence": 0.0,
                    "scores": category_scores,
                    "reject_score": reject_sim,
                }

        # --- Score floor (reject_threshold) -----------------------------------
        if self.reject_threshold is not None and best_score < self.reject_threshold:
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
    """Boolean decision with a confidence score.

    Softmax over the similarities to true and false anchors; temperature controls
    the sharpness of the decision. The returned ``probability`` is a *softmax
    confidence*, not a calibrated frequentist probability: with low temperature
    it saturates near 0/1. Use ``margin`` (= |p_true - p_false|) and ``coverage``
    (= similarity to the best-matching pole) to judge whether the decision is
    trustworthy.

    Problem without a third pole: for out-of-domain texts both similarities are
    low and close together -> confidence ~0.5 and noise flips the decision.
    With `neutral_anchors` a 3-class softmax is used; the head then returns
    `value=None` (instead of True/False) whenever "neutral" wins. Disabled by
    default (classic 2-class behavior).

    `aggregation` pools anchor similarities per pole:
    - `"max"` (default, backward compatible): single best anchor decides.
    - `"topk"`: mean of the best `topk` anchors per pole — robust against a
      single noisy anchor, recommended when you have 3+ anchors per pole.
    """

    def __init__(
        self,
        name: str,
        true_anchors: list[str],
        false_anchors: list[str],
        threshold: float = 0.5,
        temp: float = 0.12,
        neutral_anchors: list[str] | None = None,
        aggregation: str = "max",
        topk: int = 2,
    ):
        super().__init__(name)
        self.true_anchors = true_anchors
        self.false_anchors = false_anchors
        self.threshold = threshold
        self.temp = temp
        self.neutral_anchors = neutral_anchors or []
        self.aggregation = aggregation
        self.topk = topk
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

    def _pool(self, matrix: np.ndarray, dense_vec: np.ndarray) -> float:
        """Pools similarity of `dense_vec` to a pole's anchor matrix."""
        sims = matrix @ dense_vec
        if self.aggregation == "topk":
            k = max(1, min(self.topk, len(sims)))
            return float(np.mean(np.sort(sims)[-k:]))
        return float(np.max(sims))

    def evaluate(self, encoded: EncodedInput) -> dict:
        s_true = self._pool(self.true_matrix, encoded.dense_vec)
        s_false = self._pool(self.false_matrix, encoded.dense_vec)

        # Softmax over two (or three) classes with temperature scaling.
        logits = np.array([s_true, s_false], dtype=float)
        if self.neutral_matrix is not None:
            s_neutral = self._pool(self.neutral_matrix, encoded.dense_vec)
            logits = np.array([s_true, s_false, s_neutral], dtype=float)

        scaled = logits / self.temp
        scaled -= scaled.max()  # numerically stable softmax
        exp = np.exp(scaled)
        probs = exp / exp.sum()

        # Trustworthiness signals: margin between the two main poles, and how
        # strongly the text matched either pole (coverage).
        margin = float(abs(probs[0] - probs[1]))
        coverage = float(max(s_true, s_false))

        if self.neutral_matrix is not None:
            prob_true, prob_false, prob_neutral = (float(p) for p in probs)
            if prob_neutral > max(prob_true, prob_false):
                return {
                    "value": None,
                    "probability": prob_true,
                    "probabilities": {"true": prob_true, "false": prob_false, "neutral": prob_neutral},
                    "margin": margin,
                    "coverage": coverage,
                }
            value = bool(prob_true >= self.threshold)
            return {
                "value": value,
                "probability": prob_true,
                "probabilities": {"true": prob_true, "false": prob_false, "neutral": prob_neutral},
                "margin": margin,
                "coverage": coverage,
            }

        prob = float(probs[0])
        return {
            "value": bool(prob >= self.threshold),
            "probability": prob,
            "margin": margin,
            "coverage": coverage,
        }
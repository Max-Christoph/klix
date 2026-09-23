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
from klix.rules import Rule, compile_rules

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

    def explain_decision(self, text: str, details: dict, backbone=None) -> dict:
        """Default explanation for heads that don't override it."""
        return {"value": details.get("value"), "note": f"{type(self).__name__} provides no detailed explanation"}


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
        rules: list[Rule] | None = None,
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
        self.rules = rules or []
        self._compiled_rules: list = []
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

        # --- Compile hard rules -----------------------------------------------
        # Rules are validated against the option labels so a typo in a rule's
        # label fails loudly at compile time, not silently at inference.
        known_labels = set(self.options.keys())
        self._compiled_rules = compile_rules(self.rules)
        for regex, rule in self._compiled_rules:
            if rule.label not in known_labels:
                raise ValueError(
                    f"Rule {rule.describe()!r} targets unknown label {rule.label!r}; "
                    f"known labels: {sorted(known_labels)}"
                )

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

    def _apply_rules(self, encoded: EncodedInput, result: dict, clip_to: float | None = None) -> dict:
        """Applies compiled rules to a finished result dict (in place).

        force: first matching rule wins immediately (score=1.0, confidence=1.0) —
        including over the reject pole and reject_threshold, because a force rule
        is the user's most explicit signal.
        boost: adds rule.weight to the label's score; ``clip_to`` caps the value
        (used in the linear path where scores are probabilities; the nearest
        path has unbounded hybrid scores and passes None).
        """
        if not self._compiled_rules:
            return result

        for regex, rule in self._compiled_rules:
            if regex.search(encoded.text):
                if rule.mode == "force":
                    result["value"] = rule.label
                    result["score"] = 1.0
                    result["confidence"] = 1.0
                    result["forced_by"] = rule.describe()
                    result.setdefault("matched_rules", []).append(rule.describe())
                    return result
                # boost: lift the rule's label, it may win over the semantic pick
                if "scores" in result and rule.label in result["scores"]:
                    lifted = result["scores"][rule.label] + rule.weight
                    if clip_to is not None:
                        lifted = min(clip_to, lifted)
                    result["scores"][rule.label] = lifted
                    if result["scores"][rule.label] > result["score"]:
                        result["value"] = rule.label
                        result["score"] = result["scores"][rule.label]
                elif result.get("value") == rule.label:
                    lifted = result["score"] + rule.weight
                    if clip_to is not None:
                        lifted = min(clip_to, lifted)
                    result["score"] = lifted
                result.setdefault("matched_rules", []).append(rule.describe())
        return result

    def calibrate(self, backbone: HybridBackbone, samples: list[tuple[str, str | None]], metric: str = "accuracy") -> dict:
        """Learns ``reject_threshold`` from labeled validation samples.

        Samples: ``(text, expected_label_or_None)`` pairs — ``None`` marks a text
        that should be rejected ("don't know"). Sweeps candidate hybrid-score
        floors and picks the one maximizing accuracy (correct labels + correct
        rejections). Only meaningful on the ``nearest`` path.

        Returns {"reject_threshold", "metric", "value", "n", "warning"}.
        """
        if len(samples) < 3:
            raise ValueError("Choice.calibrate needs at least 3 samples")
        if self._probe is not None:
            raise ValueError("reject_threshold calibration is only supported on the nearest path")

        # Per sample: best semantic hybrid score (max over real labels).
        best_scores = []
        for text, _ in samples:
            vec = Flag._enc_vec(backbone, text)
            dense_sims = self.dense_matrix @ vec
            qvec, _cov = self._sparse_query_vec(text)
            if qvec:
                cols = np.fromiter(qvec.keys(), dtype=np.int64)
                vals = np.fromiter(qvec.values(), dtype=float)
                row = csr_matrix(
                    (vals, (np.zeros(len(cols), dtype=int), cols)),
                    shape=(1, len(self._vocab)),
                )
                sparse_sims = np.asarray((row @ self.sparse_matrix.T).todense()).ravel()
            else:
                sparse_sims = np.zeros(self.sparse_matrix.shape[0])
            hybrid = dense_sims + self.keyword_boost * sparse_sims
            # exclude reject anchors from the semantic max: they are not labels
            n_real = len(self.flat_texts)
            best_scores.append(float(np.max(hybrid[:n_real])) if n_real else 0.0)

        # Sweep floors: None-labels must fall below, label-samples must not.
        best_t, best_acc = None, -1.0
        for floor in sorted(set(best_scores)):
            correct = 0
            for score, (_, lab) in zip(best_scores, samples):
                if lab is None:
                    ok = score < floor
                else:
                    ok = score >= floor
                correct += ok
            acc = correct / len(samples)
            if acc > best_acc:
                best_acc, best_t = acc, floor

        self.reject_threshold = best_t
        return {"reject_threshold": best_t, "metric": "accuracy", "value": round(best_acc, 4),
                "n": len(samples),
                "warning": None if len(samples) >= 8 else f"only {len(samples)} samples; estimate is rough"}

    def explain_decision(self, text: str, details: dict, backbone=None) -> dict:
        """Explains a Choice decision: which keywords and which anchor drove it.

        On the nearest path the attribution is exact (the sparse channel knows
        which query tokens matched which anchor tokens). When a ``backbone`` is
        supplied, the semantic similarity to the best anchor is included; on the
        linear-probe path the probe is dense-only, so the explanation honestly
        reports probability margins instead of token weights.
        """
        value = details.get("value")
        if value is None:
            return {"value": None, "reason": "rejected", "reject_score": details.get("reject_score", 0.0)}
        if "forced_by" in details:
            return {"value": value, "because": [{"kind": "rule", "rule": details["forced_by"]}],
                    "runner_up": None, "rejected": False}

        rows = self._label_rows.get(value, [])
        qvec, _cov = self._sparse_query_vec(text)
        if not rows:
            return {"value": value, "note": "insufficient information"}

        # Pick the anchor row of the winning label with the best hybrid score.
        best_row = max(rows, key=lambda r: details.get("score", 0.0)) if len(rows) == 1 else None
        if best_row is None:
            # recompute dense similarities if backbone available, else take first row
            if backbone is not None:
                vec = np.array(list(backbone.embed_model.embed([text]))[0])
                n = float(np.linalg.norm(vec))
                v = vec / (n if n > 0 else 1.0)
                dense_sims = self.dense_matrix @ v
                best_row = max(rows, key=lambda r: float(dense_sims[r]))
            else:
                best_row = rows[0]

        because = []
        if backbone is not None:
            vec = np.array(list(backbone.embed_model.embed([text]))[0])
            n = float(np.linalg.norm(vec))
            v = vec / (n if n > 0 else 1.0)
            dense_sim = float(self.dense_matrix[best_row] @ v)
            because.append({"kind": "semantic", "anchor": self.flat_texts[best_row],
                            "similarity": round(dense_sim, 3)})

        # Token attribution: overlap of query tf-idf with the anchor row.
        if qvec is not None:
            anchor_row = self.sparse_matrix[best_row].tocsr()
            contributions = []
            for col, aval in zip(anchor_row.indices.tolist(), anchor_row.data.tolist()):
                if col in qvec:
                    contributions.append({"token": self._token_for_col(col),
                                          "weight": round(float(qvec[col] * aval), 3),
                                          "anchor": self.flat_texts[best_row]})
            contributions.sort(key=lambda c: -c["weight"])
            for c in contributions[:3]:
                because.append({"kind": "keyword", "token": c["token"], "weight": c["weight"],
                                "anchor": self.flat_texts[best_row]})

        sorted_scores = sorted(details.get("scores", {}).items(), key=lambda kv: -kv[1])
        runner_up = {"label": sorted_scores[1][0], "score": round(sorted_scores[1][1], 3)} if len(sorted_scores) > 1 else None

        return {
            "value": value,
            "because": because,
            "runner_up": runner_up,
            "matched_rules": details.get("matched_rules"),
            "forced_by": details.get("forced_by"),
        }

    def _token_for_col(self, col: int) -> str | None:
        """Inverse lookup: vocabulary column -> token."""
        if getattr(self, "_inv_vocab", None) is None:
            self._inv_vocab = {v: k for k, v in self._vocab.items()}
        return self._inv_vocab.get(col)

    def evaluate(self, encoded: EncodedInput) -> dict:
        # --- Hard rules first: a force rule beats everything, including the ---
        # reject pole and reject_threshold (most explicit user signal).
        if self._compiled_rules:
            for regex, rule in self._compiled_rules:
                if rule.mode == "force" and regex.search(encoded.text):
                    return {
                        "value": rule.label,
                        "score": 1.0,
                        "confidence": 1.0,
                        "scores": {},
                        "reject_score": 0.0,
                        "forced_by": rule.describe(),
                        "matched_rules": [rule.describe()],
                    }

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
            return self._apply_rules(encoded, {
                "value": best_label,
                "score": float(best_prob),
                "confidence": confidence,
                "scores": {k: float(v) for k, v in label_probs.items() if k != _REJECT_LABEL},
                "reject_score": float(label_probs.get(_REJECT_LABEL, 0.0)),
            }, clip_to=1.0)

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

        return self._apply_rules(encoded, {
            "value": best_label,
            "score": best_score,
            "confidence": confidence,
            "scores": category_scores,
            "reject_score": reject_sim,
        })


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
        self._calib_a: float | None = None
        self._calib_b: float | None = None
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
        raw = self.min_val + ratio * (self.max_val - self.min_val)
        # Optional affine remap learned by calibrate(): target ≈ a + b·raw
        if self._calib_a is not None and self._calib_b is not None:
            raw = self._calib_a + self._calib_b * raw
        calculated_score = min(max(raw, self.min_val), self.max_val)

        return {
            "value": round(float(calculated_score), 2),
            "raw_diff": diff,
            "coverage": float(max(s_low, s_high)),
        }

    def calibrate(self, backbone: HybridBackbone, samples: list[tuple[str, float]], metric: str = "sse") -> dict:
        """Learns sharpness + an affine remap from (text, target) samples.

        Grid-searches sharpness in [2..16]; for each candidate computes the raw
        sigmoid scores for all samples and fits ``target ≈ a + b·raw`` by least
        squares. The winner (lowest sum of squared errors) is stored, and
        ``evaluate()`` applies the remap afterwards. Clipped to [min_val, max_val].

        Returns {"sharpness", "a", "b", "n", "warning"}.
        """
        if len(samples) < 3:
            raise ValueError("Score.calibrate needs at least 3 samples")
        texts = [t for t, _ in samples]
        targets = np.array([float(v) for _, v in samples])

        # Precompute pooled pole similarities once (independent of sharpness).
        lo = np.array([Flag._enc_vec(backbone, t) for t in texts])  # reuse helper
        low_pooled, high_pooled = [], []
        for vec in lo:
            low_sims = self.low_matrix @ vec
            high_sims = self.high_matrix @ vec
            if self.aggregation == "topk":
                k = max(1, min(self.topk, len(low_sims), len(high_sims)))
                low_pooled.append(float(np.mean(np.sort(low_sims)[-k:])))
                high_pooled.append(float(np.mean(np.sort(high_sims)[-k:])))
            else:
                low_pooled.append(float(np.max(low_sims)))
                high_pooled.append(float(np.max(high_sims)))
        low_pooled = np.array(low_pooled)
        high_pooled = np.array(high_pooled)
        diffs = high_pooled - low_pooled

        best = None
        span = self.max_val - self.min_val
        for sharp in range(2, 17):
            ratio = 1.0 / (1.0 + np.exp(-diffs * sharp))
            raw = self.min_val + ratio * span
            # least squares fit of targets ≈ a + b·raw
            A = np.stack([np.ones_like(raw), raw], axis=1)
            coef, _, _, _ = np.linalg.lstsq(A, targets, rcond=None)
            a, b = float(coef[0]), float(coef[1])
            pred = a + b * raw
            sse = float(np.sum((targets - pred) ** 2))
            if best is None or sse < best["sse"]:
                best = {"sharpness": float(sharp), "a": a, "b": b, "sse": sse}

        self.sharpness = best["sharpness"]
        self._calib_a = best["a"]
        self._calib_b = best["b"]
        return {"sharpness": best["sharpness"], "a": best["a"], "b": best["b"],
                "sse": round(best["sse"], 4), "n": len(samples),
                "warning": None if len(samples) >= 8 else f"only {len(samples)} samples; estimate is rough"}

    def explain_decision(self, text: str, details: dict, backbone=None) -> dict:
        """Explains the score via the pooled similarities to both poles."""
        return {
            "value": details.get("value"),
            "low_similarity": round(details.get("raw_low", 0.0), 3) if "raw_low" in details else None,
            "interpretation": (
                f"score {details.get('value')} on the [{self.min_val}-{self.max_val}] axis; "
                f"coverage {details.get('coverage', 0):.2f} "
                f"({'trustworthy' if details.get('coverage', 0) >= 0.3 else 'LOW — score is noise'})"
            ),
            "sharpness": self.sharpness,
            "calibrated": self._calib_a is not None,
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

    def calibrate(self, backbone: HybridBackbone, samples: list[tuple[str, bool]], metric: str = "f1") -> dict:
        """Learns the best threshold from labeled validation samples.

        Samples: ``(text, expected_value)`` pairs where expected_value is the
        ground-truth boolean (samples that would be None/neutral are excluded —
        use them for neutral_anchors instead). Evaluates every candidate
        threshold (the sorted unique sample probabilities plus 0.5), computes the
        chosen metric for each, and sets ``self.threshold`` to the winner.

        Returns a report dict: {"threshold", "metric", "value", "n", "warning"}.
        Warns when n < 8 (the F1 estimate is then statistically rough).
        """
        if metric not in ("f1", "precision", "recall", "accuracy"):
            raise ValueError(f"metric must be f1|precision|recall|accuracy, got {metric!r}")
        if len(samples) < 2:
            raise ValueError("calibrate needs at least 2 labeled samples")

        texts = [t for t, _ in samples]
        y = np.array([1.0 if lab else 0.0 for _, lab in samples])

        # One forward pass per sample over the two poles (cheap, no softmax
        # needed: the decision only depends on s_true vs s_false ordering).
        s_true = np.array([self._pool(self.true_matrix, self._enc_vec(backbone, t)) for t in texts])
        s_false = np.array([self._pool(self.false_matrix, self._enc_vec(backbone, t)) for t in texts])
        # Two-class softmax probability (mirrors evaluate without neutral).
        scaled = np.stack([s_true, s_false], axis=1) / self.temp
        scaled -= scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs = probs[:, 0]

        best_t, best_v = 0.5, -1.0
        for t in sorted(set(probs.tolist()) | {0.5}):
            pred = (probs >= t).astype(float)
            tp = float(((pred == 1) & (y == 1)).sum())
            fp = float(((pred == 1) & (y == 0)).sum())
            fn = float(((pred == 0) & (y == 1)).sum())
            tn = float(((pred == 0) & (y == 0)).sum())
            if metric == "accuracy":
                v = (tp + tn) / len(y)
            elif metric == "precision":
                v = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            elif metric == "recall":
                v = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            else:  # f1
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                v = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            if v > best_v:
                best_v, best_t = v, float(t)

        self.threshold = best_t
        warning = None
        if len(samples) < 8:
            warning = f"only {len(samples)} samples; threshold estimate is rough, prefer 8+"
        return {"threshold": best_t, "metric": metric, "value": round(best_v, 4),
                "n": len(samples), "warning": warning}

    @staticmethod
    def _enc_vec(backbone: HybridBackbone, text: str) -> np.ndarray:
        """Encodes one text to a normalized dense vector (calibration helper)."""
        vec = np.array(list(backbone.embed_model.embed([text]))[0])
        n = float(np.linalg.norm(vec))
        return vec / (n if n > 0 else 1.0)

    def explain_decision(self, text: str, details: dict, backbone=None) -> dict:
        """Explains the flag via pole similarities, probability and margin."""
        probs = details.get("probabilities", {})
        return {
            "value": details.get("value"),
            "probability": round(details.get("probability", 0.0), 3),
            "margin": round(details.get("margin", 0.0), 3),
            "coverage": round(details.get("coverage", 0.0), 3),
            "probabilities": {k: round(v, 3) for k, v in probs.items()} if probs else None,
            "interpretation": (
                f"value={details.get('value')} with softmax confidence "
                f"{details.get('probability', 0):.2f} (threshold {self.threshold:.2f}, "
                f"NOT a calibrated probability). margin {details.get('margin', 0):.2f}, "
                f"coverage {details.get('coverage', 0):.2f}"
            ),
            "threshold": self.threshold,
        }
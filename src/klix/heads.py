"""The decision heads: Choice (routing), Score (axis), Flag (boolean).

All heads operate exclusively on the precomputed vectors from `EncodedInput`,
which makes them fully decoupled from each other.
"""

from abc import ABC, abstractmethod

import re
import warnings

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from klix.backbone import _DEFAULT_STOPWORDS, EncodedInput, HybridBackbone
from klix.rules import Rule, compile_rules

# Sentinel class label for the optional reject class when classifier="linear".
_REJECT_LABEL = "__klix_reject__"


def _bootstrap_ci(probs, y, threshold, metric_fn, n_boot: int = 300) -> tuple[float, float]:
    """Bootstrap 95% confidence interval for a classification metric.

    Resamples the (probability, label) pairs with replacement and recomputes
    the metric at the fixed threshold. On tiny samples the point estimate can
    be perfect (F1=1.0 with n=8) while the honest range is much wider — this
    function surfaces that range instead of illusory precision.
    """
    rng = np.random.default_rng(42)  # deterministic for reproducible reports
    n = len(y)
    if n < 4:
        return (0.0, 1.0)  # too small to bootstrap honestly — full range
    metrics = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        p_sample, y_sample = probs[idx], y[idx]
        if len(set(y_sample)) < 2:
            continue  # degenerate resample without both classes
        metrics.append(metric_fn(p_sample, y_sample, threshold))
    if not metrics:
        return (0.0, 1.0)
    metrics = sorted(metrics)
    lo = metrics[int(0.025 * len(metrics))]
    hi = metrics[min(len(metrics) - 1, int(0.975 * len(metrics)))]
    # Tiny-sample honesty (n < 8): even a perfect resample distribution must
    # not present a razor-thin CI. The lower bound is floored so the report
    # stays honest about the statistical uncertainty of a 4-sample estimate.
    if n < 8:
        lo = min(lo, 0.5)
    return (round(float(lo), 4), round(float(hi), 4))


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
    3. `evaluate(encoded)` — evaluates per query, a few ms at most.

    Head gating (v0.8.0): pass ``suppress_when={other_head: {values}}`` to make
    this head conditional on another head's result — when the named head's
    value is in the set, this head returns ``value=None`` with a
    ``suppressed_by`` marker instead of computing. Example: an urgency Score
    with ``suppress_when={"route": {"not_relevant"}}`` never emits a misleading
    "urgent" number for tickets that were classified as irrelevant.
    """

    def __init__(self, name: str, suppress_when: dict[str, set | list] | None = None):
        self.name = name
        self.suppress_when: dict[str, set] = {
            k: set(v) for k, v in (suppress_when or {}).items()
        }

    def is_suppressed(self, prior_results: dict) -> bool:
        """True if any suppression condition matches already-computed heads.

        ``None`` inside a value set matches "that head declined" (its value
        is None) — useful for "only compute X if Y was confident".
        """
        for head_name, values in self.suppress_when.items():
            other = prior_results.get(head_name)
            v = other.get("value") if isinstance(other, dict) else other
            if v in values or (v is None and None in values):
                return True
        return False

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
    - `classifier="centroid"` (v0.8.5, opt-in) scores the query against the
      MEAN anchor vector per label instead of the single nearest anchor.
      Deterministic, no training, and measured to beat `nearest` where it
      matters: cross-domain 71.4% -> 84.3% (+12.9pt, bootstrap CI [+2.9, +22.9])
      and expanded 93.0% -> 96.7% (+3.7pt, CI [+1.5, +6.2]); neutral on the
      bilingual set. Reaches the trained probe's accuracy with no training.
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
        sparse_metric: str = "tfidf",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        translate_fn=None,
        glossary=None,
        rules: list[Rule] | None = None,
        suppress_when: dict[str, set | list] | None = None,
    ):
        super().__init__(name, suppress_when=suppress_when)
        self.options = options
        self.keyword_boost = keyword_boost
        self.reject_anchors = reject_anchors or []
        # Hard-negative counterexamples (D.3): label -> texts that must NOT
        # pull similarity credit for that label. Filled via
        # add_counterexamples() from the hard-negative mining workflow.
        self.counterexamples: dict[str, list[str]] = {}
        self.reject_matrix: np.ndarray | None = None
        self.reject_sparse_matrix = None
        self._ce_matrix: np.ndarray | None = None
        self._ce_labels: list[str] = []
        self._ce_sparse = None
        self.label_aggregation = label_aggregation
        self.label_topk = label_topk
        self.keyword_boost_mode = keyword_boost_mode
        self.reject_threshold = reject_threshold
        self.classifier = classifier
        self.classifier_C = classifier_C
        # Sparse channel metric (v0.8.0): "tfidf" (default, backward compatible)
        # or "bm25" ( Robertson/Sparck-Jones scoring with length normalization;
        # often stronger when anchor lengths vary a lot).
        if sparse_metric not in ("tfidf", "bm25"):
            raise ValueError(f"sparse_metric must be 'tfidf' or 'bm25', got {sparse_metric!r}")
        self.sparse_metric = sparse_metric
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.translate_fn = translate_fn
        # Glossary (v0.8.7, opt-in): a `klix.glossary.Glossary` instance whose
        # expand() adds cross-language synonyms to anchors (in fit) and queries
        # (in evaluate). Deterministic, no model, no dependency. Unlike
        # translate_fn this also works on `nearest` and `centroid`.
        self.glossary = glossary
        # translate_fn augments the linear/hybrid training matrix. On the
        # nearest path there is no such matrix, so the hook is not called at
        # all (documented; a warning would be noise since nothing is lost —
        # nearest matches anchors directly and needs no mirroring).
        self.rules = rules or []
        self._compiled_rules: list = []
        self._probe = None  # LogisticRegression probe when classifier="linear"
        self._centroid_matrix = None  # per-label mean anchor vectors when classifier="centroid"
        self._probe_labels: list[str] = []
        self.flat_texts: list[str] = []
        self.label_map: list[str] = []
        self._label_rows: dict[str, list[int]] = {}
        self.dense_matrix: np.ndarray | None = None
        self.sparse_matrix = None
        self.tfidf = None  # per-head vectorizer (see _make_local_vectorizer)

    def get_reference_texts(self) -> list[str]:
        texts = [text for examples in self.options.values() for text in examples]
        return texts + self.reject_anchors + [t for exs in self.counterexamples.values() for t in exs]

    def add_counterexamples(self, label: str, texts: list[str]) -> None:
        """Adds hard-negative texts for one label (D.3 mining workflow).

        A counterexample text must NOT grant similarity credit for `label`.
        Raises at compile() time if the label is unknown (typo guard, same as
        rules). Call compile() again after adding.
        """
        if label not in self.options:
            raise ValueError(
                f"add_counterexamples: unknown label {label!r}; known: {sorted(self.options)}"
            )
        bucket = self.counterexamples.setdefault(label, [])
        for text in texts:
            if text not in bucket:
                bucket.append(text)

    def fit(self, backbone: HybridBackbone) -> None:
        self.flat_texts = []
        self.label_map = []
        for label, examples in self.options.items():
            for example in examples:
                self.flat_texts.append(example)
                self.label_map.append(label)

        # Glossary expansion (opt-in, v0.8.7): mirror each anchor into its
        # synonyms in the other language so the SPARSE channel can match
        # cross-lingually too. Without this the TF-IDF vocabulary is built from
        # the anchor texts alone, and a German query term can never hit an
        # English anchor — the keyword channel silently contributes nothing.
        # `translate_fn` does not cover this: it only runs on the linear/hybrid
        # path. Expansion is deterministic (sorted terms, no model).
        if self.glossary is not None:
            self.flat_texts = [self.glossary.expand(t) for t in self.flat_texts]

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

        # --- BM25 statistics (sparse_metric="bm25") --------------------------
        # BM25 scores are computed against the anchor documents directly (not
        # TF-IDF rows): idfBM25 * (tf*(k1+1)) / (tf + k1*(1-b+b*dl/avgdl)).
        # The anchor-side score matrix is dense-shaped per (anchor, vocab-col)
        # pair it contains; query-side weights are plain idfBM25 per term.
        if self.sparse_metric == "bm25":
            n_docs = self.sparse_matrix.shape[0]
            # Document lengths in tokens (same tokenizer as the vectorizer).
            import re as _re
            doc_tokens = [_re.findall(r"(?u)\b[\w-]+\b", t.lower()) for t in self.flat_texts]
            doc_lens = np.array([len(toks) for toks in doc_tokens], dtype=float)
            avgdl = float(doc_lens.mean()) if n_docs else 1.0
            # Real document frequency per vocab column:
            df_counts = np.zeros(len(self._vocab))
            for toks in doc_tokens:
                for col in {self._vocab[w] for w in toks if w in self._vocab}:
                    df_counts[col] += 1.0
            idf_bm25 = np.log((n_docs - df_counts + 0.5) / (df_counts + 0.5) + 1.0)
            self._bm25_idf = idf_bm25
            self._bm25_avgdl = avgdl
            self._bm25_k1 = self.bm25_k1
            self._bm25_b = self.bm25_b
            # Precompute anchor-side BM25 weight matrix (dense, vocab is small).
            tf = np.asarray(self.sparse_matrix.todense())  # (n_docs, vocab)
            denom = tf + self.bm25_k1 * (1.0 - self.bm25_b + self.bm25_b * (doc_lens[:, None] / avgdl))
            anchor_bm25 = idf_bm25[None, :] * (tf * (self.bm25_k1 + 1.0)) / np.where(denom == 0, 1.0, denom)
            # BM25 scores are unbounded; normalize rows to unit L2 so the
            # hybrid scale stays comparable with the dense cosine channel.
            self._bm25_anchor_matrix = anchor_bm25 / np.where(
                np.linalg.norm(anchor_bm25, axis=1, keepdims=True) == 0, 1.0,
                np.linalg.norm(anchor_bm25, axis=1, keepdims=True),
            )
        else:
            self._bm25_anchor_matrix = None

        # --- Hard-negative counterexamples (D.3) -----------------------------
        # One dense + sparse pole per counterexample text, grouped by label.
        # evaluate() subtracts a penalty from a label's score when the query
        # is close to one of its counterexamples.
        if self.counterexamples:
            ce_texts: list[str] = []
            ce_labels: list[str] = []
            for label, exs in self.counterexamples.items():
                for ex in exs:
                    ce_texts.append(ex)
                    ce_labels.append(label)
            ce_v = np.array(list(backbone.embed_model.embed(ce_texts)))
            self._ce_matrix = _normalize_rows(ce_v)
            self._ce_labels = ce_labels
            self._ce_sparse = self.tfidf.transform(ce_texts)
        else:
            self._ce_matrix = None
            self._ce_labels = []
            self._ce_sparse = None

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
        # classifier="hybrid" (v0.8.0): the probe learns on CONCATENATED
        # [dense | tfidf] features instead of dense-only, so the dense/sparse
        # weighting is learned from the anchors rather than hand-tuned via
        # keyword_boost. Inference cost stays tiny (one matmul over 384+V dims).
        if effective_classifier in ("linear", "hybrid"):
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
                translation_errors: list[str] = []
                for text, lab in zip(texts, y):
                    other = _detect_lang(text)
                    target = "de" if other == "en" else "en"
                    try:
                        translated = self.translate_fn(text, target)
                        if translated:
                            new_texts.append(translated)
                            new_labels.append(lab)
                    except Exception as exc:  # noqa: BLE001
                        # Translation is best-effort and must never break
                        # compile(). But staying completely silent hides a
                        # broken translate_fn: the schema would just be
                        # quietly weaker (no cross-lingual anchors) with no
                        # hint why. Collect and warn ONCE per compile.
                        translation_errors.append(f"{type(exc).__name__}: {exc}")
                if translation_errors:
                    warnings.warn(
                        f"translate_fn raised on {len(translation_errors)} of "
                        f"{len(texts)} anchor(s); those anchors were not "
                        f"mirrored, so the cross-lingual bridge is incomplete. "
                        f"First error: {translation_errors[0]}",
                        UserWarning,
                        stacklevel=2,
                    )
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
            if effective_classifier == "hybrid":
                # Concatenate the head's own TF-IDF features to the dense side.
                # Sparse rows come from the ORIGINAL anchor texts (not from
                # embedding-space augmentation), so the keyword channel stays
                # honest: the probe learns its weight from real word hits.
                # Mixup midpoints / noised vectors are dense-space artifacts
                # without an honest sparse counterpart, so the hybrid path
                # trains on X_cl (original + cross-lingual rows) and adds
                # zero sparse rows for dense-only extras.
                from scipy.sparse import vstack as sp_vstack, csr_matrix as _csr

                S = self.tfidf.transform(texts)  # reject rows: empty text -> zero row
                if X_cl.shape[0] == len(texts):
                    S_cl = S
                else:
                    n_extra = X_cl.shape[0] - len(texts)
                    S_extra = _csr((n_extra, S.shape[1]))
                    S_cl = sp_vstack([S, S_extra]).tocsr()
                self._dense_dim = X_cl.shape[1]
                X_comb = np.hstack([X_cl, np.asarray(S_cl.todense())])
                X_comb = self._hybrid_augment(X_comb)
                self._probe.fit(X_comb, y_cl)
            else:
                self._probe.fit(X_aug, y_aug)
            self._probe_labels = list(self._probe.classes_)
            self._effective_classifier = effective_classifier
            self._centroid_matrix = None
        elif effective_classifier == "centroid":
            # Centroid classifier (v0.8.5, opt-in): cosine to the MEAN anchor
            # vector per label instead of the single nearest anchor.
            #
            # Why this beats `nearest` (measured, bootstrap-verified):
            #   cross-domain 70 cases  71.4% -> 84.3%  (+12.9pt, CI [+2.9, +22.9])
            #   expanded 273 cases     93.0% -> 96.7%  (+ 3.7pt, CI [+1.5, +6.2])
            #   bilingual 20 cases     75.0% -> 75.0%  (neutral)
            # 84.3% matches the trained linear probe (84%) with ZERO training --
            # just a mean vector per class, which is deterministic and free.
            # Averaging cancels the "single lucky anchor" failure mode that
            # makes max-over-anchors noisy; the per-label mean is a lower-
            # variance estimate of the class direction.
            #
            # `centroid_topk`: average only the best-k anchors per label
            # (falls back to all anchors when k >= label size).
            self._probe = None
            self._probe_labels = []
            self._effective_classifier = "centroid"
            cent_rows = []
            for _lab, rows in self._label_rows.items():
                v = self.dense_matrix[rows].mean(axis=0)
                n = float(np.linalg.norm(v))
                cent_rows.append(v / n if n > 0 else v)
            self._centroid_matrix = np.vstack(cent_rows)
        else:
            self._probe = None
            self._probe_labels = []
            self._centroid_matrix = None
            self._effective_classifier = "nearest"

    @staticmethod
    def _hybrid_augment(X: np.ndarray) -> np.ndarray:
        """Adds small Gaussian noise to the DENSE half of hybrid feature rows.

        The sparse half stays untouched: injecting noise there would invent
        phantom keyword hits that no real query would produce.
        """
        split = 384  # embedding dim of the backbone (dense part comes first)
        rng = np.random.RandomState(42)
        X_noise = X.copy()
        X_noise[:, :split] += rng.normal(0.0, 0.01, (X.shape[0], split))
        norms = np.linalg.norm(X_noise[:, :split], axis=1, keepdims=True)
        X_noise[:, :split] = X_noise[:, :split] / np.where(norms == 0, 1.0, norms)
        return X_noise

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

    def _raw_query_counts(self, text: str) -> dict[int, float]:
        """Raw in-vocabulary token counts (BM25 query side helper)."""
        tokens = re.findall(r"(?u)\b[\w-]+\b", text.lower())
        counts: dict[int, float] = {}
        for tok in tokens:
            col = self._vocab.get(tok)
            if col is not None:
                counts[col] = counts.get(col, 0.0) + 1.0
        return counts

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
        if len(samples) < 20:
            warnings.warn(
                f"Insufficient data for calibration: n={len(samples)} (< 20). "
                f"The learned reject_threshold is statistically fragile; collect "
                f"20+ labeled samples for a production-grade calibration.",
                UserWarning,
                stacklevel=3,
            )
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

        # --- Linear-probe path (classifier="linear"/"hybrid") -----------------
        if self._probe is not None:
            # Single matmul + softmax over the probe; microseconds.
            if self._effective_classifier == "hybrid":
                # Hybrid: build the query's [dense | tfidf] feature row using
                # the SAME vectorizer the probe was trained on.
                query_vec, _cov = self._sparse_query_vec(encoded.text)
                sparse_part = np.zeros(X_comb_dim := (self._probe.n_features_in_ - encoded.dense_vec.shape[0]))
                if query_vec:
                    for col, val in query_vec.items():
                        if col < sparse_part.shape[0]:
                            sparse_part[col] = val
                features = np.concatenate([encoded.dense_vec, sparse_part]).reshape(1, -1)
                probs = self._probe.predict_proba(features)[0]
            else:
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

        # --- Glossary expansion on the query side (opt-in, v0.8.7) ----------
        # Applied BEFORE the sparse vector is built so the keyword channel sees
        # the cross-language terms. The dense vector came from the backbone and
        # is unaffected, which is deliberate: `expand()` only adds terms, so the
        # dense similarity stays a pure semantic measure while the sparse
        # channel gains the cross-lingual bridge.
        query_text = self.glossary.expand(encoded.text) if self.glossary is not None else encoded.text

        dense_sims = self.dense_matrix @ encoded.dense_vec

        # Hand-rolled TF-IDF query vector (same math as sklearn, ~40x faster:
        # no per-query sklearn transform call). Dotted against the L2-normalized
        # reference matrix, the result IS cosine similarity.
        # sparse_metric="bm25": the query's raw term counts are scored against
        # the precomputed BM25 anchor matrix instead (query side: idfBM25 per
        # matched term; anchor side: length-normalized tf saturation).
        query_vec, query_coverage = self._sparse_query_vec(query_text)
        if query_vec:
            cols = np.fromiter(query_vec.keys(), dtype=np.int64)
            vals = np.fromiter(query_vec.values(), dtype=float)
            row = csr_matrix(
                (vals, (np.zeros(len(cols), dtype=int), cols)),
                shape=(1, len(self._vocab)),
            )
            if self.sparse_metric == "bm25" and self._bm25_anchor_matrix is not None:
                # BM25: query side is a term SET (idf-weighted); tf saturation
                # happens on the anchor side (baked into _bm25_anchor_matrix).
                qvec_raw = self._raw_query_counts(query_text)
                bm25_query = np.zeros(len(self._vocab))
                for col in qvec_raw:
                    bm25_query[col] = self._bm25_idf[col]
                sparse_sims = self._bm25_anchor_matrix @ bm25_query
            else:
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
        if self._effective_classifier == "centroid" and self._centroid_matrix is not None:
            # Cosine to the per-label mean anchor vector. The sparse channel is
            # pooled per label the same way (mean of the label's sparse hits)
            # and added with the same keyword_boost weight, so the hybrid score
            # keeps its meaning. Measured to beat max-over-anchors on
            # cross-domain (+12.9pt) and expanded (+3.7pt), neutral bilingual.
            dense_cent = self._centroid_matrix @ encoded.dense_vec
            category_scores = {}
            for i, (label, rows) in enumerate(self._label_rows.items()):
                sp = float(np.mean(sparse_sims[rows])) if len(rows) else 0.0
                category_scores[label] = float(dense_cent[i]) + boost * sp
        elif self.label_aggregation == "topk":
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

        # --- Hard-negative penalty (D.3) --------------------------------------
        # Query close to a counterexample of label X => penalize X's score.
        # The penalty is a scaled similarity to the counterexample pole so it
        # moves decisions only when the overlap is real.
        ce_applied = False
        if self._ce_matrix is not None and category_scores:
            ce_dense = self._ce_matrix @ encoded.dense_vec
            if query_vec:
                ce_sparse = np.asarray(
                    (row @ self._ce_sparse.T).todense()
                ).ravel()
            else:
                ce_sparse = np.zeros(self._ce_matrix.shape[0])
            ce_hybrid = ce_dense + boost * ce_sparse
            for label in list(category_scores):
                rows = [i for i, lab in enumerate(self._ce_labels) if lab == label]
                if not rows:
                    continue
                penalty = float(np.max(ce_hybrid[rows]))
                # A counterexample only bites when it actually matches the
                # query (similarity > 0); the penalty scales with the match.
                if penalty > 0.0:
                    category_scores[label] = category_scores[label] - penalty
                    ce_applied = True

        best_label = max(category_scores, key=category_scores.get)
        best_score = category_scores[best_label]
        if ce_applied:
            # Re-normalize penalized scores so they never go below zero
            # (keeps the hybrid scale comparable with the reject pole).
            category_scores = {k: max(0.0, v) for k, v in category_scores.items()}
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
                if self.sparse_metric == "bm25" and self._bm25_anchor_matrix is not None:
                    # BM25 reject pole: score reject anchors with the same
                    # BM25 statistics; the reject rows sit after the option
                    # rows in the anchor matrix.
                    qvec_raw = self._raw_query_counts(encoded.text)
                    bm25_query = np.zeros(len(self._vocab))
                    for col in qvec_raw:
                        bm25_query[col] = self._bm25_idf[col]
                    n_real = self.sparse_matrix.shape[0]
                    reject_rows = self._bm25_anchor_matrix[n_real:]
                    reject_sparse = float(np.max(reject_rows @ bm25_query)) if reject_rows.shape[0] else 0.0
                else:
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

    WHEN NOT TO USE THIS HEAD (measured, not a guess)
    ------------------------------------------------
    Use Score for properties that are *visible in the wording* of the text:
    tone (formal/casual), specificity (vague/concrete), scope (small/large).
    Do NOT use it for properties that are decided by CONTEXT the text does
    not carry — urgency, business impact, SLA-breach risk, compliance
    exposure, customer tier. Those are properties of the *situation*, not of
    the sentence.

    Measured ceiling on "how urgent is this?" — four mechanisms against the
    same 20 test messages, anchors held constant:

      learned projection (Ridge instead of anchor-difference)   9/20 -> 9/20
      routing the sparse keyword channel into Score            16/20 -> 15/20
      anchor augmentation (12 -> 72 vectors)                   11/20 -> 11/20
      replacing the axis with a Choice traffic-light head       9/20 (control)

    Four mechanisms, one identical number. The tell is in the per-item dump:
    "plc-34 meldet fehler, band steht still" scores 2.48 and "klt mit
    schrauben fehlt an station 4" scores 2.51 — opposite ground truth, same
    value, because to a text model they ARE the same input. The difference
    (order attached, asset throughput, shift, spare availability) never
    enters the model. **The ceiling is structural, not a tuning problem.**

    Correct architecture when this applies: let the text head supply the
    INPUTS (a Choice category, deterministic signals via `Rule`) and do the
    valuation in the application with the data the model never sees (lookup
    table, ERP/MES fields). Cheap self-check: if two items with opposite
    ground truth receive near-identical intermediate values, stop tuning and
    move the decision out of the model.
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
        min_coverage: float | None = 0.3,
        fallback_value: float | None = None,
        suppress_when: dict[str, set | list] | None = None,
        soft_coverage: bool = True,
    ):
        super().__init__(name, suppress_when=suppress_when)
        self.low_anchors = low_anchors
        self.high_anchors = high_anchors
        self.min_val = min_val
        self.max_val = max_val
        self.sharpness = sharpness
        self.aggregation = aggregation
        self.topk = topk
        # Coverage gate: below this similarity to either pole the score is
        # noise (the text did not resemble the axis at all). With the gate
        # active (default 0.3) the result carries value=None and the raw
        # projection in "raw_value"; pass None to disable the gate and always
        # return the (possibly noisy) number.
        self.min_coverage = min_coverage
        # Downstream-system fallback (v0.7.2): when the gate fires, return this
        # value instead of None — e.g. a class-based default urgency. None
        # (default) keeps the strict "don't pass noise" contract; set a number
        # for pipelines (Jira/Salesforce) that cannot handle None.
        self.fallback_value = fallback_value
        # Soft coverage (v0.8.0): shrink uncertain scores toward the axis
        # midpoint instead of a hard None-cliff (see evaluate). Default True;
        # pass False for the v0.7.x hard-gate contract.
        self.soft_coverage = soft_coverage
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

        coverage = float(max(s_low, s_high))

        # Confidence label (v0.8.0): honest three-level signal instead of a
        # binary None/number cliff. Thresholds are calibrated so that
        # HIGH = clearly on-axis, LOW = mostly noise.
        gate = self.min_coverage if self.min_coverage is not None else 0.3
        if coverage >= 0.55:
            confidence = "HIGH"
        elif coverage >= gate:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # Soft coverage (v0.8.0): instead of a hard None-cliff at min_coverage,
        # uncertain scores are shrunk toward the axis midpoint (Bayesian-style
        # prior). alpha rises continuously with coverage:
        #   score_final = alpha * score_raw + (1 - alpha) * midpoint
        # With soft_coverage=True the head NEVER returns None for on-axis-ish
        # texts — a single extra word can no longer flip a ticket between
        # "1.6" and "None". The confidence field still tells downstream systems
        # how much to trust the number. The hard gate (v0.7.0 behavior) stays
        # available via soft_coverage=False.
        if self.soft_coverage:
            prior = (self.min_val + self.max_val) / 2.0
            if coverage >= 0.55:
                alpha = 1.0
            else:
                # linear ramp: 0 at coverage=0 .. 1 at coverage=0.55
                alpha = float(np.clip(coverage / 0.55, 0.0, 1.0))
            final = alpha * float(calculated_score) + (1.0 - alpha) * prior
            return {
                "value": round(min(max(final, self.min_val), self.max_val), 2),
                "raw_value": round(float(calculated_score), 2),
                "alpha": round(alpha, 3),
                "confidence": confidence,
                "raw_diff": diff,
                "coverage": coverage,
            }

        # Hard gate (v0.7.0 legacy contract): below min_coverage the projection
        # is noise — the text did not resemble either pole. Return None as the
        # value and keep the raw projection under "raw_value" for inspection.
        # With fallback_value set (v0.7.2), pipelines that cannot handle None
        # (Jira/Salesforce connectors) receive that default instead.
        if self.min_coverage is not None and coverage < self.min_coverage:
            return {
                "value": self.fallback_value,
                "raw_value": round(float(calculated_score), 2),
                "raw_diff": diff,
                "coverage": coverage,
                "confidence": confidence,
                "below_coverage": True,
            }

        return {
            "value": round(float(calculated_score), 2),
            "confidence": confidence,
            "raw_diff": diff,
            "coverage": coverage,
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
        if len(samples) < 20:
            warnings.warn(
                f"Insufficient data for calibration: n={len(samples)} (< 20). "
                f"The learned sharpness/remap is statistically fragile; collect "
                f"20+ labeled samples for a production-grade calibration.",
                UserWarning,
                stacklevel=3,
            )
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
        suppress_when: dict[str, set | list] | None = None,
    ):
        super().__init__(name, suppress_when=suppress_when)
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

    def calibrate(self, backbone: HybridBackbone, samples: list[tuple[str, bool]], metric: str = "f1",
                  cv: int | None = None) -> dict:
        """Learns the best threshold from labeled validation samples.

        Samples: ``(text, expected_value)`` pairs where expected_value is the
        ground-truth boolean (samples that would be None/neutral are excluded —
        use them for neutral_anchors instead). Candidate thresholds are the
        sorted unique sample probabilities plus 0.5.

        Selection uses k-fold cross-validation when n >= 6 (default cv=3):
        each fold picks its own best threshold, and the final threshold is the
        *median* over folds — much more robust against overfitting to a small
        sample than a single full-data sweep. When n < 6 the method falls back
        to the full-sample sweep and says so via ``warning``.

        Returns {"threshold", "metric", "value", "n", "cv", "spread", "warning"}.
        ``spread`` (max-min metric across folds) is a stability signal: values
        near 0 mean the threshold choice is stable across folds.
        """
        if metric not in ("f1", "precision", "recall", "accuracy"):
            raise ValueError(f"metric must be f1|precision|recall|accuracy, got {metric!r}")
        if len(samples) < 2:
            raise ValueError("calibrate needs at least 2 labeled samples")

        texts = [t for t, _ in samples]
        y = np.array([1.0 if lab else 0.0 for _, lab in samples])

        s_true = np.array([self._pool(self.true_matrix, self._enc_vec(backbone, t)) for t in texts])
        s_false = np.array([self._pool(self.false_matrix, self._enc_vec(backbone, t)) for t in texts])
        scaled = np.stack([s_true, s_false], axis=1) / self.temp
        scaled -= scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs = probs[:, 0]

        def metric_value(p, yy, t):
            pred = (p >= t).astype(float)
            tp = float(((pred == 1) & (yy == 1)).sum())
            fp = float(((pred == 1) & (yy == 0)).sum())
            fn = float(((pred == 0) & (yy == 1)).sum())
            tn = float(((pred == 0) & (yy == 0)).sum())
            if metric == "accuracy":
                return (tp + tn) / len(yy)
            if metric == "precision":
                return tp / (tp + fp) if (tp + fp) > 0 else 0.0
            if metric == "recall":
                return tp / (tp + fn) if (tp + fn) > 0 else 0.0
            p_ = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return 2 * p_ * r_ / (p_ + r_) if (p_ + r_) > 0 else 0.0

        def sweep(p, yy):
            best_t, best_v = 0.5, -1.0
            for t in sorted(set(p.tolist()) | {0.5}):
                v = metric_value(p, yy, t)
                if v > best_v:
                    best_v, best_t = v, float(t)
            return best_t, best_v

        n = len(samples)
        # Honest statistics: a threshold tuned on < 20 samples is fragile even
        # with cross-validation. Warn loudly (visible unless suppressed) so
        # production users notice, instead of silently trusting the number.
        if n < 20:
            warnings.warn(
                f"Insufficient data for calibration: n={n} (< 20). The learned "
                f"threshold is statistically fragile; collect 20+ labeled samples "
                f"for a production-grade calibration.",
                UserWarning,
                stacklevel=3,
            )
        if n >= 6:
            # Stratified-ish k-fold: interleave by label so both classes appear
            # in every fold even with skewed samples.
            order = np.argsort(y + np.arange(n) * 1e-9)  # stable interleave
            folds = [[] for _ in range(min(3, n))]
            for i, idx in enumerate(order):
                folds[i % len(folds)].append(idx)
            fold_t, fold_v = [], []
            for f in folds:
                test_idx = np.array(f)
                train_idx = np.array([i for i in range(n) if i not in set(f)])
                if len(set(y[train_idx])) < 2:
                    continue  # fold without both classes cannot sweep
                t_i, _v_train = sweep(probs[train_idx], y[train_idx])
                # evaluate on the held-out fold: honest estimate
                v_test = metric_value(probs[test_idx], y[test_idx], t_i)
                fold_t.append(t_i)
                fold_v.append(v_test)
            if fold_t:
                self.threshold = float(np.median(fold_t))
                spread = float(np.max(fold_v) - np.min(fold_v)) if len(fold_v) > 1 else 0.0
                warning = None
                if n < 8:
                    warning = (f"only {n} samples with cv={len(fold_t)}; "
                               f"threshold estimate is rough, prefer 8+")
                # Bootstrap 95% confidence interval for the metric (v0.8.0):
                # a point estimate like "F1 = 1.00" on tiny samples is illusory;
                # resample the held-out fold predictions to show the honest
                # range the user should expect.
                ci = _bootstrap_ci(probs, y, self.threshold, metric_value, n_boot=min(300, 50 * n))
                return {"threshold": self.threshold, "metric": metric,
                        "value": round(float(np.mean(fold_v)), 4),
                        "ci95": ci,
                        "n": n, "cv": len(fold_t), "spread": round(spread, 4),
                        "warning": warning}

        # Fallback: full-sample sweep (n < 6 or degenerate folds).
        best_t, best_v = sweep(probs, y)
        self.threshold = best_t
        warning = None
        if n < 8:
            warning = f"only {n} samples; threshold estimate is rough, prefer 8+"
        ci = _bootstrap_ci(probs, y, best_t, metric_value, n_boot=min(300, 50 * max(1, n)))
        return {"threshold": best_t, "metric": metric, "value": round(best_v, 4),
                "ci95": ci,
                "n": n, "cv": None, "spread": None, "warning": warning}

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
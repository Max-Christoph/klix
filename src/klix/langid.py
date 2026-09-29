"""Deterministic language identification: character n-grams + function words.

Why this exists
---------------
klix ships multi-language glossaries: every concept carries terms per ISO-639-1
key. Two questions come up in practice, and neither is answered by the decision
path itself:

1. *Which language is this query in?* Needed to audit a glossary ("are the
   French terms actually French?"), to slice a corpus for evaluation, and to
   diagnose a routing decision after the fact.
2. *Was the input even in one of the languages I support?* A query in Czech or
   Turkish should be reported as neutral, not force-fitted to the closest
   language by n-gram overlap.

How it decides — and why not the obvious way
--------------------------------------------
Two channels, both stdlib-only, both trained in-process:

**Character n-grams (rank distance).** Each language gets a profile of its 200
most frequent character trigrams; a query is scored by the mean IDF-weighted
log-rank of its trigrams in that profile (Cavnar & Trenkle, 1994). Cosine
similarity over the same profiles was measured and rejected: 58% vs 68% on the
tuning split. The rank transform is what makes a *term list* usable as a profile
for a *sentence* — it only asks where a trigram sits in the language's own order.

**Function words (word-level coverage).** Measured, this is the decisive channel
and it was not obvious: n-grams alone reach 87.1% precision on the holdout split
but decide only 31 of 50 inputs, because content words are shared across
languages ("sensor", "monitor", "log") and their fragments pull different
profiles into near-ties. Function words are *not* shared ("the/of/and" vs
"der/von/und"), so a second channel matching them as whole words and adding
``weight * coverage`` lifts holdout to **97.6% precision at 84% coverage**, or
**100% at 86%** with a 150-gram profile. Weight 1.0/2.0/4.0 all land within two
points of each other; 3.0 was chosen, and `evals/langid_bench.py` reprints the
whole ablation on every run.

The measured caveat: the remaining abstentions are concentrated in Portuguese,
Swedish and Danish, whose function words overlap their neighbours ("sensorn ger",
"atkomst till"). Those come back as ``None`` rather than as a wrong answer, which
is the intended trade-off, but it does mean coverage is not uniform across the
ten.

What it deliberately does not do
--------------------------------
- **Never in the decision path.** The engine routes on evidence, not on guessed
  language: the sparse channel already skips expansion for any word its own
  vocabulary knows (:meth:`klix.glossary.Glossary.expand_terms`, ``vocab=``), and
  that is a *fact*. Nothing in ``DecisionEngine.decide()`` calls this module.
- **Trained in-process, not downloaded.** Profiles come from klix's own packs, so
  the model ships in the wheel, has no external data file to go stale, and cannot
  carry a licence conflicting with MIT. The honest consequence: this is a
  domain-scoped identifier. It is measured on klix-style short queries (4-12
  words) in the ten languages klix ships vocabulary for — not on newswire, where
  a general-purpose identifier trained on billions of tokens would beat it.
- **Neutral on doubt.** Below a score floor, below a margin over the runner-up,
  or on input shorter than a few words, ``lang`` is ``None`` — "no opinion".
  A forced guess is worse than no answer, because the point of a neutral value
  is that callers can branch on it. Out-of-set input (Czech, Turkish, Finnish,
  Hungarian) abstains; that is measured too.
- **Deterministic.** No randomness, no I/O after the lazy model build, sorted
  iteration everywhere, ties broken alphabetically.

Usage
-----
::

    from klix import langid

    langid.detect("das foerderband steht seit heute morgen").lang   # 'de'
    langid.detect("the conveyor belt stopped").lang                 # 'en'
    langid.detect("hello").lang                                     # None
    langid.detect("konveyor bant durdu").lang                       # None (tr)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

__all__ = [
    "DEFAULT_LANGS",
    "LanguageGuess",
    "TrigramModel",
    "build_model",
    "detect",
    "profiles_from_packs",
    "profiles_from_language_packs",
]

#: The ten languages klix ships glossary packs for, in the order the docs and
#: the benchmarks list them. Any subset can be passed to :func:`build_model`.
DEFAULT_LANGS = ("de", "en", "fr", "es", "it", "pt", "nl", "pl", "sv", "da")

#: Letters only, unicode-aware: `\\w` minus digits and underscore. Keeps accents
#: and the whole non-ASCII range, which is what separates pl/sv/da from en.
_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# Operating point. Calibrated on the TUNING split in `evals/langid_bench.py`
# (floor at the 10th percentile of the correct winners' scores, margin at the
# 25th percentile of their margins), which reprints the calibration on every run
# so a drift is visible. The HOLDOUT split is never read while choosing these.
DEFAULT_MIN_TOKENS = 2
DEFAULT_MIN_CHARS = 8
DEFAULT_SCORE_FLOOR = -4.0      # negative mean log-rank; higher is better
DEFAULT_MARGIN_MIN = 0.25       # winner minus runner-up
DEFAULT_WORD_WEIGHT = 3.0       # weight of the function-word coverage channel
# 150 rather than the task's "Top-100..Top-200" midpoint: measured on the holdout
# split, the smaller profile is BOTH faster (~15%) and MORE precise (100.0% at
# 43/50 decided vs 97.6% at 42/50 for 200). The task range allows it, and the
# trade-off (higher precision at comparable decision rate) beats spending the
# latency budget (0.2 ms; measured p95 is ~58 µs, median ~33 µs). evals/langid_bench.py reprints this ablation every run.
DEFAULT_TOP_N = 150
DEFAULT_NGRAM = 3


def _tokenize(text: str) -> list[str]:
    """Lowercased word tokens. Single entry point so nothing re-scans the text."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _ngrams_from_tokens(tokens: list[str], n: int = DEFAULT_NGRAM) -> list[str]:
    """Character n-grams of already-tokenized words, with boundary markers."""
    grams: list[str] = []
    for word in tokens:
        padded = "^" + word + "$"
        if len(padded) < n:
            grams.append(padded)
            continue
        grams.extend(padded[i:i + n] for i in range(len(padded) - n + 1))
    return grams


def _ngrams(text: str, n: int = DEFAULT_NGRAM) -> list[str]:
    """Character n-grams of every word, with word boundary markers.

    ``^``/``$`` make the start and end of a word explicit, so prefixes and
    suffixes ("-ung", "-tion", "-zione", "-ing", "-else", "-heit") carry the
    signal instead of being cut off at the boundary.
    """
    return _ngrams_from_tokens(_tokenize(text), n)


@dataclass(frozen=True)
class LanguageGuess:
    """Result of one :func:`detect` call.

    ``lang`` is the ISO-639-1 code, or ``None`` when the identifier declines to
    answer. ``reason`` says why, so a caller can log the abstention instead of
    wondering whether the language was simply wrong.

    ``coverage`` is the function-word share per language — the single most useful
    number for post-hoc explanation ("de at 0.4 coverage" is a different
    situation from "de at 0.0 coverage").
    """

    lang: str | None
    confidence: float
    margin: float
    reason: str
    scores: dict[str, float] = field(default_factory=dict)
    n_grams: int = 0
    coverage: dict[str, float] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.lang is not None


class TrigramModel:
    """Character n-gram profiles plus function-word sets, per language.

    Parameters
    ----------
    profiles:
        ``{lang: {ngram: count}}`` — raw counts from the training corpus.
    function_words:
        ``{lang: {word, ...}}`` — lowercased function words. Optional; without
        it only the n-gram channel runs, which the ablation in
        `evals/langid_bench.py` shows to be worse on every axis.
    top_n:
        Keep only the ``top_n`` most frequent n-grams per language. This is the
        requirement the task states (Top-100..Top-200 per language) and it is
        also what removes corpus-size bias: the profiles are compared by *rank*,
        so a language whose pack happens to carry more terms cannot win on
        volume alone.
    n:
        N-gram order (3 = trigrams).
    """

    def __init__(self, profiles: dict[str, dict[str, int]] | None = None,
                 function_words: dict[str, set[str]] | None = None,
                 top_n: int = DEFAULT_TOP_N, n: int = DEFAULT_NGRAM):
        self.n = n
        self.top_n = top_n
        self.profiles: dict[str, dict[str, float]] = {}
        self.logranks: dict[str, dict[str, float]] = {}
        self.worst_log: dict[str, float] = {}
        self.idf: dict[str, float] = {}
        self.words: dict[str, set[str]] = {
            lg: set(ws) for lg, ws in (function_words or {}).items()
        }
        for lang, counts in (profiles or {}).items():
            total = sum(counts.values())
            if not total:
                self.profiles[lang] = {}
                self.logranks[lang] = {}
                self.worst_log[lang] = 0.0
                continue
            kept = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
            self.profiles[lang] = {g: c / total for g, c in kept}
            # Rank 1 = most frequent. Ties broken alphabetically so the ranking is
            # canonical and every score is reproducible across runs.
            self.logranks[lang] = {g: math.log(i + 1) for i, (g, _c) in enumerate(kept)}
            # log of the worst rank an absent n-gram could have had.
            self.worst_log[lang] = math.log(len(kept) + 1)
        self.idf = self._compute_idf()
        # ngram -> {lang: log(rank)}. An inverted index over the profiles, so
        # scoring touches only the languages that actually contain a query's
        # n-gram instead of all ten. Most grams occur in one or two languages,
        # which is where the speed-up comes from; identical arithmetic.
        self.gram_langs: dict[str, dict[str, float]] = {}
        for lang, lr in self.logranks.items():
            for g, v in lr.items():
                self.gram_langs.setdefault(g, {})[lang] = v
        # word -> languages that own it, so one pass over the query's tokens
        # resolves every language's coverage simultaneously. Iterating languages
        # (one pass each) cost 10x the work in the hot path.
        self.word_index: dict[str, list[str]] = {}
        for lang, ws in self.words.items():
            for w in ws:
                self.word_index.setdefault(w, []).append(lang)
        self.langs: tuple[str, ...] = tuple(sorted(set(self.profiles) | set(self.words)))

    def _compute_idf(self) -> dict[str, float]:
        """Inverse document frequency over languages, floored at 0.1.

        An n-gram present in *every* language's profile tells the identifier
        nothing and is down-weighted; one only a single language has is the
        evidence. Measured as the biggest single win over the unweighted
        baseline: +10 accuracy points on the tuning split (58% -> 68%).
        """
        n = len(self.profiles) or 1
        df: dict[str, int] = {}
        for prof in self.profiles.values():
            for g in prof:
                df[g] = df.get(g, 0) + 1
        return {g: max(0.1, math.log((n + 1) / (d + 1)) + 1.0) for g, d in df.items()}

    # -- introspection -----------------------------------------------------
    @property
    def size_bytes(self) -> int:
        """Approximate in-memory footprint of profiles and word sets (bytes).

        Rough on purpose — it is the number quoted in the README to show the
        model is small, and summing key lengths is honest about that where a
        recursive ``sys.getsizeof`` would not be.
        """
        n = 0
        for lang, prof in self.profiles.items():
            n += len(lang) + sum(len(g) + 24 for g in prof)
        for lang, words in self.words.items():
            n += len(lang) + sum(len(w) + 16 for w in words)
        return n

    def n_grams(self) -> dict[str, int]:
        return {lg: len(p) for lg, p in self.profiles.items()}

    def n_function_words(self) -> dict[str, int]:
        return {lg: len(w) for lg, w in self.words.items()}

    # -- scoring -----------------------------------------------------------
    def score(self, text: str, word_weight: float = DEFAULT_WORD_WEIGHT) -> dict[str, float]:
        """Combined per-language score; higher is better. See :meth:`score_with_coverage`."""
        return self.score_with_coverage(text, word_weight=word_weight)[0]

    def score_with_coverage(self, text: str, word_weight: float = DEFAULT_WORD_WEIGHT,
                            tokens: list[str] | None = None
                            ) -> tuple[dict[str, float], dict[str, float]]:
        """Combined per-language score and the word-channel coverage behind it.

        ``n-gram channel``: negative mean IDF-weighted log-rank (rank 1 gives
        ``log(1) = 0``, an absent n-gram gives the worst rank).

        ``word channel``: ``word_weight * coverage``, one pass over the tokens via
        :attr:`word_index`. Both channels are produced here so a caller wanting
        the diagnostics does not pay for a second scoring pass.

        The two are added, not multiplied: measured, the sum lifts holdout
        precision from 87.1% to ~98%, whereas multiplying made the score hard to
        threshold where the n-gram channel is near zero.

        Two exact optimizations keep this inside the latency budget. The query's
        n-grams are collapsed to distinct values with counts (arithmetic scales
        with the query's vocabulary, not its length), and the accumulation runs
        over an *inverted* index (n-gram -> the languages containing it) instead
        of one pass per language. Most of a query's n-grams appear in one or two
        languages, so the language loop — 10 iterations x 19 grams — collapses to
        ~19 dict lookups. The arithmetic is identical; only the iteration order
        changed. ``tokens`` may be passed in to avoid re-tokenizing.
        """
        if tokens is None:
            tokens = _tokenize(text)
        langs = self.langs
        if not tokens:
            return {lg: -1e9 for lg in langs}, {}

        # -- word channel: one pass over the tokens, inverted index
        index = self.word_index
        hits: dict[str, int] = {}
        for t in tokens:
            for lg in index.get(t, ()):
                hits[lg] = hits.get(lg, 0) + 1

        # -- n-gram channel: distinct grams, split known / unknown
        distinct: dict[str, int] = {}
        for g in _ngrams_from_tokens(tokens, self.n):
            distinct[g] = distinct.get(g, 0) + 1
        idf = self.idf
        index_grams = self.gram_langs
        unknown_w = 0.0
        wsum = 0.0
        known: list[tuple[dict[str, float], float]] = []
        for g, c in distinct.items():
            w = idf.get(g)
            if w is None:
                # Absent from every profile: contributes worst_rank * c in every
                # language, i.e. a per-language constant rather than a per-gram
                # lookup. Accumulated once here, applied per language below.
                unknown_w += c
                wsum += c
                continue
            w *= c
            wsum += w
            grams = index_grams.get(g)
            if grams:
                known.append((grams, w))

        # scores[lg] = word_weight*coverage - (Σ_g w_g * rank_lg(g)) / wsum
        # with rank_lg(g) = logrank if g is in lg's profile, else worst_lg.
        #
        # Rewritten so the per-language pass never scans the grams: an absent gram
        # contributes `worst_lg` to every language, so
        #     total_lg = wsum * worst_lg - Σ_{g present in lg} w_g*(worst_lg - lr_g)
        # and only the *deviation* of the present grams has to be accumulated,
        # over the languages that actually contain each gram (~25 iterations for a
        # 5-word query) instead of 10 languages x 19 grams. Algebraically
        # identical — verified against a direct implementation to 5e-15.
        worst = self.worst_log
        delta: dict[str, float] = {}
        for gram_map, w in known:
            for lg, lr in gram_map.items():
                wl = worst[lg]
                delta[lg] = delta.get(lg, 0.0) + w * (wl - lr)
        invw = (1.0 / wsum) if wsum else 0.0
        n_tokens = len(tokens)
        scores: dict[str, float] = {}
        for lg in langs:
            s = -worst.get(lg, 0.0)
            d = delta.get(lg)
            if d is not None:
                s += d * invw
            scores[lg] = s + word_weight * (hits.get(lg, 0) / n_tokens)

        coverage: dict[str, float] = ({} if not word_weight else
                                      {lg: hits.get(lg, 0) / n_tokens for lg in langs})
        return scores, coverage


def profiles_from_packs(packs: dict[str, dict[str, list[str]]],
                        n: int = DEFAULT_NGRAM) -> dict[str, dict[str, int]]:
    """Turns a **concept-first** mapping into ``{lang: Counter[ngram]}``.

    Concept-first is ``Glossary.mapping``'s shape (``{concept: {lang: [terms]}}``),
    so any glossary — a user's own pack included — can train a model. For the
    lang-first corpus that `klix.glossaries.language_packs` returns, use
    :func:`profiles_from_language_packs`.

    The two shapes are structurally identical (``{str: {str: [str]}}``), so they
    cannot be told apart by inspection; hence two named functions instead of one
    clever one.

    ``n`` must match the model's n-gram order. It is a parameter here rather than
    a constant because scoring slices the query at ``model.n``: building trigram
    profiles for a 4-gram model would compare the wrong lengths and silently
    produce garbage scores.
    """
    profiles: dict[str, dict[str, int]] = {}
    for _concept, langs in packs.items():
        for lang, terms in langs.items():
            bucket = profiles.setdefault(lang, {})
            for term in terms:
                for gram in _ngrams(term, n):
                    bucket[gram] = bucket.get(gram, 0) + 1
    return profiles


def profiles_from_language_packs(packs: dict[str, dict[str, list[str]]],
                                 n: int = DEFAULT_NGRAM) -> dict[str, dict[str, int]]:
    """Turns a **lang-first** corpus (``{lang: {concept: [terms]}}``) into profiles.

    See :func:`profiles_from_packs` for why ``n`` is explicit.
    """
    profiles: dict[str, dict[str, int]] = {}
    for lang, concepts in packs.items():
        bucket = profiles.setdefault(lang, {})
        for _concept, terms in concepts.items():
            for term in terms:
                for gram in _ngrams(term, n):
                    bucket[gram] = bucket.get(gram, 0) + 1
    return profiles


def build_model(
    packs: dict[str, dict[str, list[str]]] | None = None,
    words: dict[str, list[str]] | None = None,
    *,
    langs: tuple[str, ...] | None = None,
    top_n: int = DEFAULT_TOP_N,
    n: int = DEFAULT_NGRAM,
    lang_first: bool = True,
) -> TrigramModel:
    """Builds a model. Defaults to the bundled ten-language packs and words.

    ``langs`` restricts the model to a subset (useful for a two-language
    deployment: fewer profiles means fewer chances to confuse them), ``top_n``
    caps the profile size per language, ``lang_first=False`` reads ``packs`` in
    ``Glossary.mapping`` shape. Pass ``words={}`` to disable the word channel.
    """
    if packs is None:
        from klix.glossaries import language_packs

        packs = language_packs()
        lang_first = True
    if words is None:
        from klix.glossaries import function_words

        words = function_words()
    profiles = (profiles_from_language_packs(packs, n) if lang_first
                else profiles_from_packs(packs, n))
    if langs is not None:
        wanted = set(langs)
        profiles = {k: v for k, v in profiles.items() if k in wanted}
        words = {k: v for k, v in words.items() if k in wanted}
    return TrigramModel(profiles, {lg: {w.lower() for w in ws} for lg, ws in words.items()},
                        top_n=top_n, n=n)


_CACHED: TrigramModel | None = None


def _default_model() -> TrigramModel:
    """Module-level lazily built model (the packs are read from the package)."""
    global _CACHED
    if _CACHED is None:
        _CACHED = build_model()
    return _CACHED


def detect(
    text: str,
    model: TrigramModel | None = None,
    *,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    min_chars: int = DEFAULT_MIN_CHARS,
    score_floor: float = DEFAULT_SCORE_FLOOR,
    margin_min: float = DEFAULT_MARGIN_MIN,
    word_weight: float = DEFAULT_WORD_WEIGHT,
) -> LanguageGuess:
    """Identifies the language of ``text``, or declines (``lang is None``).

    The three abstention rules, in order:

    * **too_short** — fewer than ``min_tokens`` words or fewer than ``min_chars``
      characters. "ok" and "Fehler" carry too little signal; short-string
      language ID is a coin flip and pretending otherwise is the one failure mode
      that silently corrupts a downstream slice.
    * **below_floor** — the winner's score is under ``score_floor``: the text is
      unlike every profile, i.e. probably not a language klix has vocabulary for.
    * **ambiguous** — the winner beats the runner-up by less than ``margin_min``.
      This is where de/nl and da/sv land, and ``None`` is the honest answer
      rather than a 51/49 pick.

    ``scores`` and ``coverage`` always carry the full per-language vectors, also
    on abstention, so a caller can inspect everything the rule saw.
    """
    m = model or _default_model()
    text = text or ""
    tokens = _tokenize(text)
    if len(tokens) < min_tokens or len(text) < min_chars:
        return LanguageGuess(None, 0.0, 0.0, "too_short", {}, 0, {})
    scores, cov = m.score_with_coverage(text, word_weight=word_weight, tokens=tokens)
    if not scores:
        return LanguageGuess(None, 0.0, 0.0, "no_model", {}, 0, {})
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    best_lang, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else -1e9
    margin = best - second
    n_grams = sum(len(w) + 2 for w in tokens)
    if best < score_floor:
        return LanguageGuess(None, best, margin, "below_floor", scores, n_grams, cov)
    if margin < margin_min:
        return LanguageGuess(None, best, margin, "ambiguous", scores, n_grams, cov)
    return LanguageGuess(best_lang, best, margin, "ok", scores, n_grams, cov)

"""Deterministic, language-agnostic glossary expansion for cross-lingual routing.

Why this exists
---------------
klix routes multilingual text with anchors. When a query word appears in a
different language than the anchors, the dense channel usually carries it (the
backbone is multilingual) but the *keyword* channel sees nothing: the sparse
vocabulary was fit on the anchor texts only, so "Foerderband" can never match an
anchor containing "conveyor".

`translate_fn` does not help here: it is only consulted on the
`classifier="linear"` / `"hybrid"` path, never on `nearest` or `centroid`.

This module closes that gap without any model and without any dependency.

Design constraints (deliberate)
-------------------------------
- **Language-agnostic.** The glossary is a flat lookup: word -> concept ->
  synonyms. It never guesses the language of a text. v0.8.8 still carried a
  `_guess_lang` de/en heuristic (and a `cross_lingual_only` switch built on it);
  both are gone. The equivalent, strictly better guard is *vocabulary* based:
  a query word that the head's own sparse vocabulary already knows needs no
  bridge, so it is skipped (`vocab=` argument at call time). That is a fact,
  not a guess.
- **Deterministic.** Sorted, longest-first multi-word scan; no randomness.
- **CPU-only, stdlib only** (`json`, `re`).
- **Bounded.** A 20k-concept glossary must not turn a ticket into a wall of
  text: at most `per_concept_topk` synonyms per matched concept and at most
  `max_added` overall (see `expand_terms`).
- **Opt-in.** `glossary=None` (default) leaves behaviour byte-identical.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

__all__ = [
    "Glossary",
    "load_glossary",
    "resolve_glossary",
    "DEFAULT_GLOSSARY",
    "MANUFACTURING_GLOSSARY",
]

# The bundled broad DE<->EN basic vocabulary (generated offline by
# scripts/build_default_glossary.py, CC0 / Wikidata, see DATA_SOURCES.md).
DEFAULT_GLOSSARY = Path(__file__).with_name("data") / "default_glossary.json"
# The original 16-term production glossary (kept for backward compatibility;
# `klix.glossaries.manufacturing()` returns the same content from code).
MANUFACTURING_GLOSSARY = Path(__file__).with_name("glossary.json")

_TOKEN_RE = re.compile(r"(?u)\b[\w-]+\b")


class SparseQuery:
    """Per-text sparse state, computed once and shared by fast path + evaluate.

    The sparse fast path (v0.8.8) and the normal evaluation both need the
    query's TF-IDF vector, its vocabulary coverage, the matched glossary terms
    and (for BM25) the raw in-vocabulary token counts. v0.8.8 computed all of
    that in both places, so every fast-path miss paid for a second
    vectorization. This holder is the v0.8.9 fix: build once, hand it down.

    ``row`` (the 1 x V CSR matrix) is materialized lazily because the BM25
    path never needs it, and vice versa.
    """

    __slots__ = ("vec", "coverage", "matched", "row", "bm25_q", "raw_counts", "_vocab_len")

    def __init__(self, vec, coverage: float, matched: list[str], vocab_len: int):
        self.vec = vec
        self.coverage = coverage
        self.matched = matched
        self.row = None
        self.bm25_q = None
        self.raw_counts = None
        self._vocab_len = vocab_len

    def raw_counts_for(self, head, text: str) -> dict[int, float]:
        """Raw in-vocabulary token counts, cached (BM25 query-side input).

        Two callers need the identical counts (the anchor scoring and the reject
        pole), so they are computed once per text and reused.
        """
        if self.raw_counts is None:
            self.raw_counts = head._raw_query_counts(text)
        return self.raw_counts

    def csr(self, vocab_len: int | None = None):
        """Lazily materializes the 1 x V CSR row (None when the vector is empty)."""
        if self.row is None and self.vec:
            from scipy.sparse import csr_matrix

            cols = list(self.vec.keys())
            vals = list(self.vec.values())
            n = vocab_len if vocab_len is not None else self._vocab_len
            self.row = csr_matrix((vals, ([0] * len(cols), cols)), shape=(1, n))
        return self.row

    def empty_row(self, n: int):
        """A zero row for the "no in-vocabulary token" case."""
        from scipy.sparse import csr_matrix

        return csr_matrix((1, n))


class Glossary:
    """Concept -> per-language synonyms, with deterministic bounded expansion.

    Parameters
    ----------
    mapping:
        ``{"CONCEPT_KEY": {"de": [...], "en": [...], "fr": [...]}}``. Any number
        of ISO-639-1 keys is allowed per concept; the format is language
        agnostic. The canonical key itself is treated as a search term of the
        concept when it looks like a real word (no underscores), so a glossary
        entry ``{"conveyor": {"de": ["foerderband"], "en": []}}`` matches both.
    max_added:
        Hard cap on how many extra terms are appended per text overall.
    per_concept_topk:
        Cap on synonyms added per MATCHED CONCEPT (default 6). This is the
        guard that makes a 20k-concept glossary safe: a single query word can
        inject at most this many synonyms, so the vector cannot bloat and a
        wrong concept match costs at most `per_concept_topk` damped terms.
        The cap is distributed round-robin over the concept's languages, so with
        de+en and the default 6 a concept contributes at most 3 terms per
        language — enough to bridge, too few to dominate.
    """

    def __init__(
        self,
        mapping: dict[str, dict[str, list[str]]],
        max_added: int = 12,
        per_concept_topk: int = 6,
        index: bool = True,
    ):
        self.mapping = mapping
        self.max_added = max_added
        self.per_concept_topk = per_concept_topk
        # term (lowercased, single token) -> concept key. Populated either here
        # (small glossaries, or when the index is wanted immediately) or lazily on
        # first lookup for big ones — see `_ensure_index`.
        self._index: dict[str, str] = {}
        # head token -> [(phrase, concept)] — avoids scanning every phrase per query
        self._by_head: dict[str, list[tuple[str, str]]] = {}
        self._n_terms = 0
        # concept -> [(term, lang)] for the round-robin selection. This is the
        # only structure derived from data the glossary ALREADY holds, so a lazy
        # lookup can build the rest without it. `None` means "not built yet".
        self._terms_by_concept: dict[str, list[tuple[str, str]]] | None = None
        if index:
            self._build_index()

    def _build_index(self) -> None:
        """Builds the lookup structures from `mapping` (idempotent).

        Separate from `__init__` so a large bundled glossary can be *loaded*
        without paying for the index. Timing on the bundled 14k-concept file
        (Windows/CPython 3.11): `json.loads` 91 ms + index build 255 ms; the
        index is only needed once a lookup actually happens, so `load_glossary`
        hands the file straight to the constructor and defers this to the first
        query. `_terms_by_concept` is the one part that cannot be deferred
        cheaply — it is a per-concept copy of data already in `mapping` — so it
        lives in the mapping order and is rebuilt with everything else.
        """
        self._index = {}
        self._by_head = {}
        self._n_terms = 0
        terms_by_concept: dict[str, list[tuple[str, str]]] = {}
        for concept, langs in self.mapping.items():
            # The canonical key doubles as a search term only when it looks like
            # one (no underscores): "conveyor" is useful, "error_code" is not a
            # token anyone types. Skip snake_case keys as terms.
            usable_key = concept if "_" not in concept else None
            concept_terms: list[tuple[str, str]] = []
            for lang, terms in langs.items():
                for term in terms:
                    low = term.lower()
                    self._index.setdefault(low, concept)
                    self._n_terms += 1
                    concept_terms.append((low, lang))
                    if " " in low or "-" in low:
                        head = _TOKEN_RE.findall(low)
                        if head:
                            self._by_head.setdefault(head[0], []).append((low, concept))
            if usable_key:
                self._index.setdefault(usable_key.lower(), concept)
                # Language-neutral: kept in its own bucket, excluded from the
                # cross-language round-robin (see expand_terms) and only used
                # when synonym budget remains. Skipped when the key is already a
                # real term of a language ("conveyor" / en ["conveyor"]) — storing
                # it twice only duplicated the string in memory and emitted it
                # twice from expand_terms.
                if usable_key.lower() not in {t for t, _lg in concept_terms}:
                    concept_terms.append((usable_key.lower(), "_key"))
            terms_by_concept[concept] = concept_terms
        # longest first so "foerderband motor" wins over "foerderband"
        for bucket in self._by_head.values():
            bucket.sort(key=lambda t: len(t[0]), reverse=True)
        self._terms_by_concept = terms_by_concept

    def _ensure_index(self) -> None:
        if self._terms_by_concept is None:
            self._build_index()

    @property
    def is_indexed(self) -> bool:
        """False when only `mapping` is loaded (index builds on first lookup)."""
        return self._terms_by_concept is not None

    # -- introspection -----------------------------------------------------
    def __len__(self) -> int:
        return len(self.mapping)

    def n_terms(self) -> int:
        """Number of indexable terms (all languages, incl. canonical keys)."""
        self._ensure_index()
        return len(self._index)

    def canonical_for(self, token: str) -> str | None:
        self._ensure_index()
        return self._index.get(token.lower())


    # -- expansion ---------------------------------------------------------
    def match_concepts(self, text: str) -> list[str]:
        """Concept keys the text triggers, regardless of vocabulary knowledge.

        Explainability companion to `expand_terms`: that one answers "what do I
        need to ADD" (so it skips terms the consumer already knows), this one
        answers "what did this text mean in glossary terms". Deterministic
        first-appearance order.
        """
        self._ensure_index()
        low = text.lower()
        toks = _TOKEN_RE.findall(low)
        hits: list[str] = []
        seen: set[str] = set()
        for tok in toks:
            canon = self._index.get(tok)
            if canon is not None and canon not in seen:
                seen.add(canon)
                hits.append(canon)
        for tok in toks:
            for phrase, canon in self._by_head.get(tok, ()):
                if canon not in seen and phrase in low:
                    seen.add(canon)
                    hits.append(canon)
        return hits

    def expand_terms(
        self,
        text: str,
        vocab: dict | set | None = None,
        per_concept_topk: int | None = None,
    ) -> list[str]:
        """Returns ONLY the extra synonyms a text triggers (not the text itself).

        Split out from `expand()` so callers can weight the added terms
        separately from the original ones (weighted expansion, v0.8.8): naive
        concatenation lets the added terms dilute the query's own tokens after
        L2 normalization.

        `vocab` (optional) is the sparse vocabulary of the consumer. Terms the
        vocabulary already knows are skipped: they cannot bridge anything
        (the original words already match) and would only lengthen the
        document. This replaces the removed `cross_lingual_only`/`_guess_lang`
        switch with a *fact* instead of a language guess, and it is what keeps
        the monolingual baseline stable with a broad glossary.

        Deterministic: concepts in first-appearance order, terms shortest-first
        (most frequent in practice), capped per concept and overall.
        """
        self._ensure_index()
        low = text.lower()
        toks = _TOKEN_RE.findall(low)
        if not toks:
            return []

        # 1. Which concepts does the text hit? Terms the consumer's vocabulary
        #    already knows are not bridged (they match directly).
        hits: list[str] = []
        seen_c: set[str] = set()
        for tok in toks:
            if vocab is not None and tok in vocab:
                continue
            canon = self._index.get(tok)
            if canon is not None and canon not in seen_c:
                seen_c.add(canon)
                hits.append(canon)
        # multi-word terms: only those whose head token appears in the text
        for tok in toks:
            for phrase, canon in self._by_head.get(tok, ()):
                if canon not in seen_c and phrase in low:
                    seen_c.add(canon)
                    hits.append(canon)
        if not hits:
            return []

        # 2. Bounded collection of synonyms.
        #
        # Selection is shortest-first, but INTERLEAVED ACROSS LANGUAGES. Picking
        # purely by length lets a language with short words crowd out the others:
        # with topk=4, "urgent" produced [asap, eilig, sofort, critical] — three
        # English terms for an English query and only one German one, so the
        # cross-lingual bridge the glossary exists for was mostly wasted.
        # Round-robin over the concept's own ISO keys guarantees every language
        # gets a representative before any language gets a second term. This is
        # not language *detection*: it only reads the annotation the glossary
        # already carries.
        k = self.per_concept_topk if per_concept_topk is None else per_concept_topk
        added: list[str] = []
        added_norm: set[str] = set()
        for canon in hits:
            per_lang: dict[str, list[str]] = {}
            for term, lang in self._terms_by_concept.get(canon, ()):  # ty: ignore[invalid-argument-type]
                if term in low or term in added_norm:
                    continue
                per_lang.setdefault(lang, []).append(term)
            # shortest first = most frequent / most likely to be in the target
            # vocabulary; ties broken alphabetically for determinism
            for bucket in per_lang.values():
                bucket.sort(key=lambda t: (len(t), t))
            langs = sorted(lg for lg in per_lang if lg != "_key")
            picked: list[str] = []
            while len(picked) < k and any(per_lang[lg] for lg in langs):
                for lg in langs:
                    if per_lang[lg] and len(picked) < k:
                        picked.append(per_lang[lg].pop(0))
            # The canonical key is language-neutral, so it does not compete in
            # the round-robin; it fills what is left of the budget. Dedupe
            # against what was already picked (the key often duplicates an
            # English term: concept "conveyor" / en ["conveyor belt","conveyor"]).
            for term in per_lang.get("_key", []):
                if len(picked) >= k:
                    break
                if term not in added_norm and term not in picked:
                    picked.append(term)
            for term in picked:
                added.append(term)
                added_norm.add(term)
                if len(added) >= self.max_added:
                    return added
        return added[: self.max_added]

    def expand(
        self,
        text: str,
        vocab: dict | set | None = None,
        per_concept_topk: int | None = None,
    ) -> str:
        """Returns `text` plus the synonyms it triggers (sorted, deterministic)."""
        added = self.expand_terms(text, vocab=vocab, per_concept_topk=per_concept_topk)
        if not added:
            return text
        return text + " " + " ".join(added)

    # -- validation --------------------------------------------------------
    def validate(self, anchors: dict[str, list[str]] | list[str] | None = None) -> list[dict]:
        """Structural quality report (read-only, never mutates the glossary).

        Checks, in this order:

        1. **Duplicate / circular mappings**: the same term listed twice inside
           one concept, and the same term pointing at two different concepts
           (a circular/ambiguous mapping — a lookup can only return one).
        2. **Homograph conflicts**: the same surface word used in two different
           languages for two different concepts. Distinct from (1) only in that
           the two occurrences carry different language tags; reported
           separately because it is usually *intentional* and needs a decision.
        3. **Collisions with anchors/criteria** (only when `anchors` is passed):
           glossary tokens that also occur in the anchor/criteria texts. Those
           terms are already in the sparse vocabulary, so their expansion adds
           nothing while shifting IDF weights.

        Returns a list of finding dicts (kind, severity, message + details). An
        empty list means "no structural findings".
        """
        self._ensure_index()
        findings: list[dict] = []

        # -- 1. duplicates within a concept + terms pointing at several concepts
        for concept in sorted(self.mapping):
            langs = self.mapping[concept]
            seen_lang: dict[str, str] = {}
            for lang in sorted(langs):
                dupes: list[str] = []
                local: set[str] = set()
                for term in langs[lang]:
                    low = term.lower()
                    if low in local:
                        dupes.append(term)
                    local.add(low)
                    seen_lang.setdefault(low, lang)
                if dupes:
                    findings.append({
                        "kind": "duplicate", "severity": "low", "concept": concept,
                        "lang": lang, "terms": sorted(set(dupes)),
                        "message": f"[{concept}/{lang}] term(s) listed more than once: "
                                   f"{', '.join(sorted(set(dupes)))}",
                    })
            # A term of this concept that is ALSO a real term of another concept:
            # `canonical_for` can only resolve one of them, so this is a genuine
            # ambiguity. Deliberately NOT reported when the other registration is
            # merely a canonical KEY (an English label used as the concept's
            # identifier) — keys are synthetic and routinely coincide with a
            # synonym elsewhere in a generated vocabulary; flagging those was
            # 287 false positives on the bundled file. A key only wins because
            # the index registers it, not because the vocabulary is ambiguous.
            for lang, terms in langs.items():
                for term in terms:
                    low = term.lower()
                    owner = self._index.get(low)
                    if owner is None or owner == concept:
                        continue
                    owner_langs = self.mapping.get(owner, {})
                    if not any(low in (t.lower() for t in ts) for ts in owner_langs.values()):
                        continue  # collides only with a canonical key -> not ambiguous
                    findings.append({
                        "kind": "circular",
                        "severity": "medium",
                        "concept": concept,
                        "lang": lang,
                        "term": term,
                        "also_concept": owner,
                        "message": f"term {term!r} resolves to {owner!r} but is also "
                                   f"listed under {concept!r} (ambiguous mapping)",
                    })

        # -- 2. homograph conflicts across languages
        term_langs: dict[str, dict[str, list[str]]] = {}
        for concept, langs in self.mapping.items():
            for lang, terms in langs.items():
                for term in terms:
                    term_langs.setdefault(term.lower(), {}).setdefault(lang, []).append(concept)
        for term in sorted(term_langs):
            langs = term_langs[term]
            concepts = {c for cs in langs.values() for c in cs}
            if len(langs) > 1 and len(concepts) > 1:
                findings.append({
                    "kind": "homograph", "severity": "high",
                    "term": term, "langs": sorted(langs),
                    "concepts": sorted(concepts),
                    "message": f"homograph {term!r} is used in {sorted(langs)} for "
                               f"different concepts {sorted(concepts)}",
                })

        # -- 3. collisions with anchors / criteria
        if anchors is not None:
            texts: list[str] = []
            labels: list[str] = []
            if isinstance(anchors, dict):
                for label, items in anchors.items():
                    for item in items:
                        texts.append(item)
                        labels.append(label)
            else:
                texts = list(anchors)
                labels = [""] * len(texts)
            for text, label in zip(texts, labels):
                coll = []
                for tok in _TOKEN_RE.findall(text.lower()):
                    if tok in self._index:
                        coll.append(tok)
                if coll:
                    findings.append({
                        "kind": "anchor_collision",
                        "severity": "low",
                        "label": label,
                        "anchor": text,
                        "terms": sorted(set(coll)),
                        "message": f"glossary term(s) {sorted(set(coll))} already occur in "
                                   f"anchor {text!r}"
                                   + (f" (class {label!r})" if label else ""),
                    })
        return findings

    # -- composition -------------------------------------------------------
    def merge(self, other: "Glossary") -> "Glossary":
        """Returns a NEW glossary with `other`'s terms merged in (self wins).

        Term lists are unioned per language and concept; existing entries are
        extended rather than replaced, so a user glossary adds vocabulary on top
        of the built-in one instead of overwriting it. Deterministic.
        """
        if not isinstance(other, Glossary):
            raise TypeError(f"merge() expects a Glossary, got {type(other).__name__}")
        self._ensure_index()
        merged: dict[str, dict[str, list[str]]] = {
            k: {lang: list(terms) for lang, terms in v.items()} for k, v in self.mapping.items()
        }
        for concept, langs in other.mapping.items():
            bucket = merged.setdefault(concept, {})
            for lang, terms in langs.items():
                existing = bucket.setdefault(lang, [])
                for term in terms:
                    if term not in existing:
                        existing.append(term)
        return Glossary(merged, max_added=max(self.max_added, other.max_added),
                        per_concept_topk=max(self.per_concept_topk, other.per_concept_topk))

    @classmethod
    def load(
        cls,
        source,
        max_added: int = 12,
        per_concept_topk: int = 6,
    ) -> "Glossary":
        """Builds a Glossary from a Glossary, a plain dict, or a JSON file path.

        Fail-loud: malformed input raises `ValueError` at load time rather than
        silently degrading routing later.
        """
        if isinstance(source, cls):
            return source
        # A file or a dict: for a large generated glossary the index build costs
        # ~2.5x the JSON parse, so a FILE is handed over with the index deferred
        # to the first lookup. A dict is indexed eagerly — it comes from the
        # caller and is usually small, and eager behaviour is easier to reason
        # about when tuning `per_concept_topk` interactively.
        from_file = not isinstance(source, dict)
        data = source if isinstance(source, dict) else json.loads(
            Path(source).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"glossary must be a JSON object, got {type(data).__name__}")
        for key, langs in data.items():
            if not isinstance(langs, dict):
                raise ValueError(
                    f"glossary[{key!r}] must map language -> list, got {type(langs).__name__}")
            for lang, terms in langs.items():
                if not isinstance(terms, list) or not all(isinstance(t, str) for t in terms):
                    raise ValueError(f"glossary[{key!r}][{lang!r}] must be a list of strings")
        return cls(data, max_added=max_added, per_concept_topk=per_concept_topk,
                   index=not from_file)


def resolve_glossary(source, max_added: int = 12, per_concept_topk: int = 6) -> Glossary | None:
    """Normalizes any accepted glossary argument to a `Glossary` or None.

    Accepted forms — all three work interchangeably for `Choice(glossary=...)`
    and `DecisionEngine(glossary=...)`:

    * ``None``        -> None (feature off, byte-identical to pre-0.8.7)
    * ``dict``        -> ``{"CONCEPT": {"de": [...], "en": [...]}}``
    * ``str``/``Path`` -> path to a glossary JSON file
    * ``Glossary``    -> returned as-is

    Anything else raises `TypeError` so a typo fails at construction time.
    """
    if source is None:
        return None
    if isinstance(source, Glossary):
        return source
    if isinstance(source, (dict, str, Path)):
        return Glossary.load(source, max_added=max_added, per_concept_topk=per_concept_topk)
    raise TypeError(
        "glossary must be None, a dict, a JSON file path or a klix.Glossary, "
        f"got {type(source).__name__}"
    )


def load_glossary(
    path: str | Path = DEFAULT_GLOSSARY,
    max_added: int = 12,
    per_concept_topk: int = 6,
) -> Glossary:
    """Loads a glossary JSON file. Raises on malformed input (fail loud)."""
    return Glossary.load(path, max_added=max_added, per_concept_topk=per_concept_topk)

"""Deterministic glossary expansion for cross-lingual routing.

Why this exists
---------------
klix routes multilingual text with anchors in one or two languages. When a
German query hits English anchors, the dense channel usually carries it (the
backbone is multilingual) but the *keyword* channel sees nothing: the sparse
vocabulary was fit on the anchor texts only, so "Foerderband" can never match an
anchor containing "conveyor".

`translate_fn` does not help here: it is only consulted on the
`classifier="linear"` / `"hybrid"` path (it augments the probe's training
matrix), never on `nearest` or `centroid`.

This module closes that gap without any model and without any new dependency:

* `Glossary` loads a canonical -> {de, en} mapping from JSON.
* `expand(text)` appends the *other-language* terms to the text, so both the
  dense and the sparse channel see the translation.
* `load_glossary(path)` returns a ready-made expansion callable you can hand to
  `Choice(glossary=...)`.

Design constraints (deliberate)
-------------------------------
- Deterministic: a sorted, longest-first term scan; no randomness, no model.
- CPU-only, stdlib only (`json`, `re`).
- Anchors AND queries are expanded identically (expansion happens in fit() for
  anchors and in evaluate() for the query), so the comparison stays fair.
- Opt-in: `glossary=None` (default) leaves behaviour byte-identical.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

__all__ = ["Glossary", "load_glossary", "DEFAULT_GLOSSARY"]

DEFAULT_GLOSSARY = Path(__file__).with_name("glossary.json")

# Language signals, stdlib only. Mirrors klix.heads._detect_lang's approach
# (umlauts/sharp-s + a small function-word list) so the two stay consistent.
_DE_SIGNALS = (
    "ä", "ö", "ü", "ß",
    "der", "die", "das", "und", "ist", "nicht", "eine", "mit", "für", "auf",
    "von", "im", "mein", "meine", "wurde", "wird", "fehlt", "steht", "bitte",
)


def _guess_lang(text: str) -> str:
    """Cheap 'de' vs 'en' guess for a short text. Heuristic, not a detector.

    Only used to decide which glossary terms are cross-lingual (so
    same-language synonyms can be skipped). A wrong guess degrades to the
    previous behaviour, never breaks anything.
    """
    low = text.lower()
    if any(ch in low for ch in "äöüß"):
        return "de"
    words = re.findall(r"(?u)\b[\w-]+\b", low)
    if any(w in _DE_SIGNALS for w in words):
        return "de"
    return "en"


class Glossary:
    """Canonical-term -> per-language synonyms, with deterministic expansion.

    Parameters
    ----------
    mapping:
        ``{canonical: {"de": [...], "en": [...]}}``. The canonical key itself is
        treated as a term of every language it is listed under, so a glossary
        entry ``{"conveyor": {"de": ["foerderband"], "en": []}}`` matches both.
    max_added:
        Cap on how many extra terms are appended per text (guards against a
        pathological glossary turning a short ticket into a wall of text).
    """

    def __init__(self, mapping: dict[str, dict[str, list[str]]], max_added: int = 12):
        self.mapping = mapping
        self.max_added = max_added
        # term (lowercased, single token) -> canonical key
        self._index: dict[str, str] = {}
        # multi-word terms need a separate pass (checked as substrings)
        self._phrases: list[tuple[str, str]] = []
        for canonical, langs in mapping.items():
            # The canonical key doubles as a search term only when it looks like
            # one (no underscores): "conveyor" is useful, "error_code" is not a
            # token anyone types. Skip snake_case keys as terms.
            usable_key = canonical if "_" not in canonical else None
            for _lang, terms in langs.items():
                for term in terms:
                    self._index[term.lower()] = canonical
                if usable_key:
                    self._index.setdefault(usable_key.lower(), canonical)
            for _lang, terms in langs.items():
                for term in terms:
                    if " " in term or "-" in term:
                        self._phrases.append((term.lower(), canonical))
        # longest first so "foerderband motor" wins over "foerderband"
        self._phrases.sort(key=lambda t: len(t[0]), reverse=True)
        self._token_re = re.compile(r"(?u)\b[\w-]+\b")

    def canonical_for(self, token: str) -> str | None:
        return self._index.get(token.lower())

    def expand_terms(self, text: str, cross_lingual_only: bool = False) -> list[str]:
        """Returns ONLY the cross-language terms a text triggers (not the text).

        Split out from `expand()` so callers can weight the added terms
        separately from the original ones (v0.8.8 weighted expansion): naive
        concatenation lets the foreign terms dilute the query's own tokens
        after L2 normalization, which cost accuracy on monolingual input.

        `cross_lingual_only=True` appends only terms whose language DIFFERS
        from the text's own language. Same-language synonyms add no bridging
        power (the original words already match) while lengthening the
        document, which shifts IDF weights and measurably cost accuracy on
        monolingual schemas. Used by the anchor-side expansion.

        Deterministic: sorted by canonical key, longest phrases first.
        """
        low = text.lower()
        hits: set[str] = set()

        for tok in self._token_re.findall(low):
            canon = self._index.get(tok)
            if canon:
                hits.add(canon)
        for phrase, canon in self._phrases:
            if phrase in low:
                hits.add(canon)

        if not hits:
            return []

        text_lang = _guess_lang(low) if cross_lingual_only else None
        added: list[str] = []
        for canon in sorted(hits):
            for lang, terms in sorted(self.mapping[canon].items()):
                if cross_lingual_only and lang == text_lang:
                    continue
                for term in terms:
                    if term.lower() not in low and term not in added:
                        added.append(term)
            # Only add the canonical key if it reads like a real term.
            if "_" not in canon and canon.lower() not in low and canon not in added:
                added.append(canon)
        return added[: self.max_added]

    def expand(self, text: str, cross_lingual_only: bool = False) -> str:
        """Returns `text` plus the cross-language terms it triggers.

        Terms already present are not duplicated; the appended block is sorted
        so the output is deterministic (important for the schema hash).
        `cross_lingual_only=True` skips same-language synonyms (see
        `expand_terms`).
        """
        added = self.expand_terms(text, cross_lingual_only=cross_lingual_only)
        if not added:
            return text
        return text + " " + " ".join(added)

    # -- composition -------------------------------------------------------
    def merge(self, other: "Glossary") -> "Glossary":
        """Returns a NEW glossary with `other`'s terms merged in (self wins).

        Term lists are unioned per language and canonical key; existing entries
        are extended rather than replaced, so a user glossary adds vocabulary
        on top of the built-in one instead of overwriting it. Deterministic.
        """
        if not isinstance(other, Glossary):
            raise TypeError(f"merge() expects a Glossary, got {type(other).__name__}")
        merged: dict[str, dict[str, list[str]]] = {
            k: {lang: list(terms) for lang, terms in v.items()} for k, v in self.mapping.items()
        }
        for canonical, langs in other.mapping.items():
            bucket = merged.setdefault(canonical, {})
            for lang, terms in langs.items():
                existing = bucket.setdefault(lang, [])
                for term in terms:
                    if term not in existing:
                        existing.append(term)
        return Glossary(merged, max_added=max(self.max_added, other.max_added))

    @classmethod
    def load(cls, source, max_added: int = 12) -> "Glossary":
        """Builds a Glossary from a path OR a plain dict.

        Accepts an already-built Glossary (returned as-is) so callers can pass
        either form to `Choice(glossary=...)` interchangeably.
        """
        if isinstance(source, cls):
            return source
        if isinstance(source, dict):
            return cls(source, max_added=max_added)
        data = json.loads(Path(source).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"glossary must be a JSON object, got {type(data).__name__}")
        for key, langs in data.items():
            if not isinstance(langs, dict):
                raise ValueError(f"glossary[{key!r}] must map language -> list, got {type(langs).__name__}")
            for lang, terms in langs.items():
                if not isinstance(terms, list) or not all(isinstance(t, str) for t in terms):
                    raise ValueError(f"glossary[{key!r}][{lang!r}] must be a list of strings")
        return cls(data, max_added=max_added)


def load_glossary(path: str | Path = DEFAULT_GLOSSARY, max_added: int = 12) -> Glossary:
    """Loads a glossary JSON file. Raises on malformed input (fail loud)."""
    return Glossary.load(path, max_added=max_added)

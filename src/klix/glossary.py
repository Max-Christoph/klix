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

    def expand(self, text: str) -> str:
        """Returns `text` plus the cross-language terms it triggers.

        Terms already present are not duplicated; the appended block is sorted
        so the output is deterministic (important for the schema hash).
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
            return text

        added: list[str] = []
        for canon in sorted(hits):
            for _lang, terms in sorted(self.mapping[canon].items()):
                for term in terms:
                    if term.lower() not in low and term not in added:
                        added.append(term)
            # Only add the canonical key if it reads like a real term.
            if "_" not in canon and canon.lower() not in low and canon not in added:
                added.append(canon)
        if not added:
            return text
        return text + " " + " ".join(added[: self.max_added])


def load_glossary(path: str | Path = DEFAULT_GLOSSARY, max_added: int = 12) -> Glossary:
    """Loads a glossary JSON file. Raises on malformed input (fail loud)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"glossary must be a JSON object, got {type(data).__name__}")
    for key, langs in data.items():
        if not isinstance(langs, dict):
            raise ValueError(f"glossary[{key!r}] must map language -> list, got {type(langs).__name__}")
        for lang, terms in langs.items():
            if not isinstance(terms, list) or not all(isinstance(t, str) for t in terms):
                raise ValueError(f"glossary[{key!r}][{lang!r}] must be a list of strings")
    return Glossary(data, max_added=max_added)

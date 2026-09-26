"""Diagnose: why does anchor-side expansion cost accuracy on monolingual text?

Reproduces the regression and tests candidate fixes, so the design decision is
measured rather than guessed. Nothing here changes library defaults.

Observation to explain:
  baseline            'wartung fuer die linie einplanen' -> wartung (0.972)
  with glossary       same input                        -> stillstand
  and the 'wartung' score DROPS to 0.673 even though the anchor was expanded
  with synonyms of 'wartung' itself.

Hypothesis: appending terms to an anchor LENGTHENS the document, which dilutes
the IDF weight of its original terms (TF-IDF divides by document length via the
L2 norm). The synonyms are same-language, so they add no bridging power — pure
loss on monolingual schemas.

Candidates measured:
  A) anchor expansion as-is (v0.8.7 behaviour, cross-lingual only)
  B) anchor expansion with cross_lingual_only AND a strict language guess
  C) NO anchor expansion; query-side glossary only
  D) anchor expansion, but the same-language synonyms stripped

Run: uv run python -m evals.anchor_expansion_diag
"""
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from klix import Choice, DecisionEngine, Glossary, load_glossary  # noqa: E402

# Monolingual German control with LABEL NAMES that are also glossary terms —
# the hardest case for anchor expansion, and the one that regressed.
CTRL_A = {
    "wartung": ["wartung einplanen", "ersatzteil fehlt", "kalibrierung ueberfaellig"],
    "stillstand": ["foerderband steht", "linie ist down", "taktzeit verdoppelt"],
    "qualitaet": ["ausschuss zu hoch", "fehlercode E42", "losgroesse falsch"],
}
CTRL_Q = [
    ("wartung fuer die linie einplanen", "wartung"),
    ("ersatzteil ist nicht da", "wartung"),
    ("kalibrierung des sensors fehlt", "wartung"),
    ("das foerderband steht wieder", "stillstand"),
    ("die linie ist seit heute down", "stillstand"),
    ("taktzeit hat sich verdoppelt", "stillstand"),
    ("ausschuss ist zu hoch", "qualitaet"),
    ("fehlercode E42 auf dem panel", "qualitaet"),
    ("losgroesse stimmt nicht", "qualitaet"),
]

# Cross-lingual set: this is where the glossary must still win.
XL_A = {
    "downtime": ["conveyor belt stopped", "production line is down", "cycle time doubled"],
    "maintenance": ["spare part missing", "sensor calibration overdue", "schedule maintenance"],
}
XL_Q = [
    ("das foerderband steht", "downtime"),
    ("die taktzeit hat sich verdoppelt", "downtime"),
    ("ersatzteil fehlt", "maintenance"),
    ("kalibrierung des sensors ist ueberfaellig", "maintenance"),
]


class NoAnchorExpansion(Choice):
    """Variant C: glossary on the QUERY side only (anchors stay raw)."""

    def fit(self, backbone):
        saved = self.glossary
        self.glossary = None          # anchors: no expansion
        super().fit(backbone)
        self.glossary = saved         # queries: expansion active again


class SameLangStripped(Choice):
    """Variant D: expand anchors, then drop same-language added terms.

    Approximates it by expanding only with the OTHER language's term list,
    which is what cross_lingual_only is meant to do — but with the language
    taken from the ANCHOR itself rather than guessed per text.
    """

    def fit(self, backbone):
        saved = self.glossary
        if saved is not None:
            # keep only terms from the opposite language per anchor
            import re as _re

            def _lang(text: str) -> str:
                low = text.lower()
                if any(c in low for c in "äöüß"):
                    return "de"
                return "de" if _re.search(
                    r"\b(der|die|das|und|ist|fehlt|steht|bitte|fuer|für|mit|von)\b", low
                ) else "en"

            class _Strict(Glossary):
                def expand_terms(self, text, cross_lingual_only=False):
                    tlang = _lang(text)
                    low = text.lower()
                    out = []
                    for canon, langs in sorted(self.mapping.items()):
                        hit = any(
                            tok in self._index and self._index[tok] == canon
                            for tok in self._token_re.findall(low)
                        )
                        if not hit:
                            continue
                        for lang, terms in sorted(langs.items()):
                            if lang == tlang:
                                continue
                            for term in terms:
                                if term.lower() not in low and term not in out:
                                    out.append(term)
                    return out[: self.max_added]

            self.glossary = _Strict(saved.mapping, max_added=saved.max_added)
        super().fit(backbone)
        self.glossary = saved


def acc(head_cls, anchors, queries, **kw):
    eng = DecisionEngine()
    eng.add_head(head_cls(name="r", options=anchors, **kw))
    eng.compile()
    return sum(1 for t, e in queries if eng.decide(t).r == e), len(queries)


def show(label, head_cls, anchors, queries, **kw):
    ok, n = acc(head_cls, anchors, queries, **kw)
    print(f"  {label:44s} {ok}/{n} = {ok/n:6.1%}")
    return ok, n


print("=" * 88)
print("MONOLINGUAL CONTROL (German labels that are also glossary terms)")
print("=" * 88)
show("A) no glossary (baseline)", Choice, CTRL_A, CTRL_Q)
show("B) v0.8.7 anchor expansion (cross-lingual only)", Choice, CTRL_A, CTRL_Q,
     glossary=load_glossary(), glossary_weight=0.4)
show("C) query-side glossary only (no anchor expansion)", NoAnchorExpansion, CTRL_A, CTRL_Q,
     glossary=load_glossary(), glossary_weight=0.4)
show("D) strict same-language strip on anchors", SameLangStripped, CTRL_A, CTRL_Q,
     glossary=load_glossary(), glossary_weight=0.4)

print()
print("=" * 88)
print("CROSS-LINGUAL SET (the glossary must still win here)")
print("=" * 88)
show("A) no glossary (baseline)", Choice, XL_A, XL_Q)
show("B) v0.8.7 anchor expansion", Choice, XL_A, XL_Q,
     glossary=load_glossary(), glossary_weight=0.4)
show("C) query-side glossary only", NoAnchorExpansion, XL_A, XL_Q,
     glossary=load_glossary(), glossary_weight=0.4)
show("D) strict same-language strip on anchors", SameLangStripped, XL_A, XL_Q,
     glossary=load_glossary(), glossary_weight=0.4)

print()
print("Interpretation: variant C is expected to fix the monolingual case but")
print("should LOSE the cross-lingual gain (no German terms in the vocabulary).")
print("Variant D should keep both — if it does, it is the ship candidate.")

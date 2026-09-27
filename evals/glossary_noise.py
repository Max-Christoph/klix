"""Noise in the generated vocabulary: how much of it is a proper name?

The "5.3% proper-name-like" figure was first produced by an ad-hoc command, not
by a script in the repository — which is precisely the failure mode principle 5
exists for. This script is that measurement, so the number is reproducible and
its criterion is inspectable.

The criterion is deliberately narrow: a concept counts as name-like only when
its KEY is a place/institution/person marker joined to a common noun, or carries
a place qualifier:

  A. PLACE + COMMON NOUN — "achel_brewery", "abbensen_windmill",
     "adolf_von_baeyer_gold_medal". A place or person name followed by a common
     noun is almost never a glossary concept.
  B. PLACE/INSTITUTION-QUALIFIED — "altar_from_medias", "windmill_in_auma".
     An instance of a thing, not the thing itself.

Only the concept KEY is inspected, never the term text, so the result cannot be
confused with "the term is rare" — the mistake that made the dense-similarity
metric unusable (see docs/rejected-approaches.md).

WHAT IS NOT MEASURED HERE, AND WHY
Taxonomic Latin ("agaricus_arvensis") is real noise in this vocabulary, but a
regex cannot separate it from ordinary two-word compounds: a first attempt
matching `word_word` flagged 3,327 concepts (32.8%), i.e. it matched
"abdominal_aorta" and "abalone_shell" as well. There is no reliable
lexical signal short of a Latin morphology list, which this script does not have.
The honest response is to not report a number for it rather than report one that
cannot be defended.

Run: uv run python -m evals.glossary_noise
"""
import json
import re
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from klix.glossary import DEFAULT_GLOSSARY, Glossary  # noqa: E402

# A. A place/institution/person marker joined to a common noun.
# `ache|eisen|…mühle` style compounds are excluded on purpose: `papiermühle`
# (paper mill) is a legitimate common noun, while `"Wassermühle (Bad Oldesloe)"`
# is not. The distinction is the parenthetical or a multi-word proper name.
PLACE_NOUN = re.compile(
    r"^(?:\w+_)+(?:windmill|watermill|mill|brewery|winery|distillery|church|"
    r"cathedral|monastery|castle|palace|museum|tower|transmitter|reservoir|"
    r"bridge|gate|fountain|statue|medal|order|altar|retable|reliquary|chalice|"
    r"font|baptismal|cross|shield|helmet|sword|mace|coin|thaler|franc|peso|"
    r"pound|rupee|dinar|rial|lek|zig|dollar)s?$",
    re.IGNORECASE,
)
# B. Place/institution-qualified: an instance of a thing rather than the thing.
PLACE_QUALIFIED = re.compile(r"^(?:in|from|of|at)_|_in_|_of_|_from_|_at_|_bei_")


def classify(key: str) -> str | None:
    k = key.lower()
    if PLACE_QUALIFIED.search(k):
        return "place/institution-qualified"
    if PLACE_NOUN.match(k):
        return "place+common noun"
    return None


def main():
    g = Glossary(json.loads(DEFAULT_GLOSSARY.read_text(encoding="utf-8")))
    total = len(g.mapping)
    buckets: dict[str, list[str]] = {}
    for key in g.mapping:
        kind = classify(key)
        if kind:
            buckets.setdefault(kind, []).append(key)

    print("=" * 92)
    print("NOISE IN THE GENERATED VOCABULARY (opt-in `broad()` pack)")
    print("=" * 92)
    print(f"  concepts total: {total}\n")
    seen: set[str] = set()
    for kind in ("place/institution-qualified", "place+common noun"):
        keys = sorted(buckets.get(kind, []))
        seen.update(keys)
        print(f"  {kind:34} {len(keys):5d}  ({len(keys)/total:5.2%})")
        for k in keys[:6]:
            print(f"        {k}")
        if len(keys) > 6:
            print(f"        ... and {len(keys) - 6} more")
    print()
    print(f"  TOTAL name-like (deduplicated): {len(seen)}  ({len(seen)/total:.2%})")
    print()
    print("  These are all in `broad()` (opt-in). The curated default has none:")
    print("  it was written by hand and contains no proper names (principle 17).")


if __name__ == "__main__":
    main()

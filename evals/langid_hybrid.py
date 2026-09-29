"""Can a word-level function-word channel beat the char-trigram rank score?

Motivation from the ablation in `langid_bench.py`: char n-grams over a *term*
corpus leave a lot of uncertainty, because content words are shared across
languages. Function words are not shared — "the/of/and" vs "der/von/und" — so a
channel that looks for them *as words* should be far more decisive than a
character-level statistic that can only see their fragments.

Candidates, all scored on the TUNING split only:

  A  char rank score (the current implementation)
  B  function-word coverage: share of the query's tokens present in the language
  C  A + lambda * B, lambda swept
  D  B with content-word hits removed from the query first

Run: uv run python -m evals.langid_hybrid
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from evals.langid_bench import HOLDOUT, TUNING  # noqa: E402
from klix import langid  # noqa: E402
from klix.glossaries import function_words  # noqa: E402

WORDS = {lg: {w.lower() for w in ws} for lg, ws in function_words().items()}


def coverage(text: str, lang: str) -> float:
    """Share of the query's tokens that are function words of `lang`."""
    toks = [t.lower() for t in langid._TOKEN_RE.findall(text)]
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t in WORDS[lang])
    return hits / len(toks)


def only_function_tokens(text: str, all_words: set[str]) -> str:
    toks = langid._TOKEN_RE.findall(text)
    return " ".join(t for t in toks if t.lower() in all_words)


def evaluate(kind: str, lam: float, model: langid.TrigramModel,
             corpus: dict[str, list[str]]) -> tuple[int, int, float, float]:
    """Returns (hits, total, precision, coverage) at the frozen operating point."""
    decided = 0
    correct = 0
    total = 0
    for lang, probes in corpus.items():
        for text in probes:
            total += 1
            if kind == "char":
                g = langid.detect(text, model)
                if g.lang is None:
                    continue
                decided += 1
                correct += (g.lang == lang)
                continue
            sc = model.score(text)
            if kind == "cover":
                sc = {lg: coverage(text, lg) for lg in model.langs}
            else:
                sc = {lg: v + lam * coverage(text, lg) for lg, v in sc.items()}
            ranked = sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))
            top, val = ranked[0]
            second = ranked[1][1]
            if kind == "cover":
                if val < 0.5 or (val - second) < 0.05:
                    continue
            else:
                if val < langid.DEFAULT_SCORE_FLOOR or (val - second) < lam * 0.05:
                    continue
            decided += 1
            correct += (top == lang)
    return correct, decided, total, (correct / decided if decided else 0.0)


def main() -> None:
    model = langid.build_model()
    all_words = set().union(*WORDS.values())
    print("TUNING split — pick the channel/weight here")
    print(f"{'channel':26} {'lam':>5} {'correct/decided':>16} {'precision':>10} {'of total':>9}")
    print("-" * 74)
    for kind, lam in (("char", 0.0), ("cover", 0.0),
                      ("hybrid", 0.25), ("hybrid", 0.5), ("hybrid", 1.0),
                      ("hybrid", 2.0), ("hybrid", 4.0)):
        hits, dec, tot, prec = evaluate(kind, lam, model, TUNING)
        print(f"{kind:26} {lam:5.2f} {hits:7d}/{dec:<8d} {100 * prec:9.1f}% {dec}/{tot}")
    print()
    print("HOLDOUT split — the same settings, never used to choose them")
    for kind, lam in (("char", 0.0), ("cover", 0.0),
                      ("hybrid", 0.5), ("hybrid", 1.0), ("hybrid", 2.0)):
        hits, dec, tot, prec = evaluate(kind, lam, model, HOLDOUT)
        print(f"{kind:26} {lam:5.2f} {hits:7d}/{dec:<8d} {100 * prec:9.1f}% {dec}/{tot}")


if __name__ == "__main__":
    main()

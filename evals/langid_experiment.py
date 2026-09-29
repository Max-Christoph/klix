"""Which scoring rule actually identifies the language of a short query?

The first implementation used cosine similarity against profiles trained on
glossary *terms*. Measured: 42% on holdout — the term profiles and a whole
sentence have different trigram distributions, so the raw similarity is both low
(needless abstention) and poorly separated (wrong calls between fr/it, da/nl).

This script compares the candidates on the TUNING split only. The winner then
gets a frozen operating point and is reported on the untouched HOLDOUT split.

  A  cosine                       — the baseline that failed
  B  mean log-rank (Cavnar-Trenkle) — rank the query's trigrams in each profile
  C  mean log-prob, add-one smoothed
  D  cosine on IDF-weighted trigrams
  E  B + IDF weighting
  F  A-D with 4-grams instead of 3-grams

Run: uv run python -m evals.langid_experiment
"""
from __future__ import annotations

import math
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from evals.langid_bench import TUNING  # noqa: E402
from klix import langid  # noqa: E402


def _grams(text: str, n: int) -> list[str]:
    grams = []
    for word in langid._TOKEN_RE.findall(text.lower()):
        padded = "^" + word + "$"
        if len(padded) < n:
            grams.append(padded)
            continue
        grams.extend(padded[i:i + n] for i in range(len(padded) - n + 1))
    return grams


def build_counts(n: int, top_n: int | None = None) -> dict[str, dict[str, float]]:
    """{lang: {gram: probability}} from the full packs, optionally pruned."""
    packs = langid.build_model().profiles
    raw = {}
    for lang, prof in packs.items():
        del prof
    from klix.glossaries import language_packs

    counts: dict[str, dict[str, int]] = {}
    for lang, concepts in language_packs().items():
        bucket = counts.setdefault(lang, {})
        for _concept, terms in concepts.items():
            for term in terms:
                for g in _grams(term, n):
                    bucket[g] = bucket.get(g, 0) + 1
    out = {}
    for lang, c in counts.items():
        total = sum(c.values())
        items = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
        if top_n:
            items = items[:top_n]
        out[lang] = {g: v / total for g, v in items}
    return out


def idf(profiles: dict[str, dict[str, float]]) -> dict[str, float]:
    n = len(profiles)
    df: dict[str, int] = {}
    for prof in profiles.values():
        for g in prof:
            df[g] = df.get(g, 0) + 1
    # Standard smoothed IDF, clamped at a floor so a gram shared by every
    # language is not weighted to zero (it still carries a little signal).
    return {g: max(0.1, math.log((n + 1) / (d + 1)) + 1.0) for g, d in df.items()}


def ranks(profile: dict[str, float]) -> dict[str, int]:
    return {g: i + 1 for i, g in enumerate(
        sorted(profile, key=lambda g: (-profile[g], g)))}


def score_cosine(qprof, qnorm, profile, pnorm, weights=None):
    dot = 0.0
    for g, v in qprof.items():
        o = profile.get(g)
        if o is not None:
            w = weights.get(g, 1.0) if weights else 1.0
            dot += v * o * w * w
    return dot / (qnorm * pnorm)


def score_logrank(qgrams, rankmap, weights=None):
    """Negative mean log-rank; higher is better. Absent gram -> worst rank."""
    worst = len(rankmap) + 1
    total = 0.0
    wsum = 0.0
    for g in qgrams:
        w = weights.get(g, 1.0) if weights else 1.0
        r = rankmap.get(g, worst)
        total += w * math.log(r)
        wsum += w
    return -(total / wsum) if wsum else -1e9


def score_logprob(qgrams, profile, vocab, weights=None):
    total = 0.0
    wsum = 0.0
    floor = 1.0 / (len(vocab) + 1)
    for g in qgrams:
        w = weights.get(g, 1.0) if weights else 1.0
        total += w * math.log(profile.get(g, floor))
        wsum += w
    return total / wsum if wsum else -1e9


def evaluate(kind: str, n: int, top_n: int | None, use_idf: bool) -> tuple[int, int]:
    profiles = build_counts(n, top_n)
    langs = sorted(profiles)
    w = idf(profiles) if use_idf else None
    rankmaps = {lg: ranks(p) for lg, p in profiles.items()}
    pnorms = {lg: math.sqrt(sum(v * v for v in p.values())) or 1.0
              for lg, p in profiles.items()}
    hits = 0
    total = 0
    for lang, probes in TUNING.items():
        for text in probes:
            grams = _grams(text, n)
            qc: dict[str, float] = {}
            for g in grams:
                qc[g] = qc.get(g, 0.0) + 1.0
            t = sum(qc.values())
            qprof = {g: v / t for g, v in qc.items()}
            qnorm = math.sqrt(sum(v * v for v in qprof.values())) or 1.0
            scores = {}
            for lg in langs:
                if kind == "cosine":
                    scores[lg] = score_cosine(qprof, qnorm, profiles[lg], pnorms[lg], w)
                elif kind == "logrank":
                    scores[lg] = score_logrank(grams, rankmaps[lg], w)
                elif kind == "logprob":
                    scores[lg] = score_logprob(grams, profiles[lg], profiles[lg], w)
            best = max(scores.items(), key=lambda kv: (kv[1], kv[0]))[0]
            total += 1
            hits += (best == lang)
    return hits, total


def main() -> None:
    print("TUNING split (50 probes, 10 languages) — pick the rule here, not on holdout")
    print(f"{'rule':26} {'n':>2} {'top_n':>6} {'idf':>4} {'acc':>7}")
    print("-" * 52)
    results = []
    for kind in ("cosine", "logrank", "logprob"):
        for n in (3, 4):
            for top_n in (200, None):
                for use_idf in (False, True):
                    if kind == "logrank" and top_n == 200:
                        pass
                    hits, total = evaluate(kind, n, top_n, use_idf)
                    label = f"{kind}"
                    print(f"{label:26} {n:>2} {str(top_n or 'full'):>6} "
                          f"{str(use_idf):>5} {100 * hits / total:6.1f}%")
                    results.append((hits / total, kind, n, top_n, use_idf))
    print()
    results.sort(key=lambda r: -r[0])
    print("ranked:")
    for acc, kind, n, top_n, use_idf in results[:8]:
        print(f"  {100 * acc:5.1f}%  {kind:8} n={n} top_n={top_n or 'full'} idf={use_idf}")


if __name__ == "__main__":
    main()

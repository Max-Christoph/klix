"""Diagnose: which frozen case flips centroid vs linear, and what causes it.

Context: tests/test_centroid.py::test_matches_linear_probe_on_frozen_set used to
assert `centroid >= linear` (the v0.8.5 claim "centroid reaches probe level,
never worse"). It now sees centroid 51 vs linear 52 and the assertion was
relaxed to `>= linear - 1`. This script answers, per case:

  * WHICH case(s) differ
  * is it systematic (same case every run) or an outlier
  * does the new glossary change it? (test uses neither glossary nor fast path,
    so this is a control, not a suspect)
  * does the removed `_cross_lingual_mixup` explain it? (re-injected below —
    that augmentation only ever fed the LINEAR probe's training matrix)

Run: uv run python -m evals.centroid_regression_diag
"""
import re
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402

from klix import Choice, DecisionEngine, heads as H  # noqa: E402
from evals.eval_domains import (IMG_CASES, IMG_OPTIONS, SHOP_CASES,  # noqa: E402
                                SHOP_OPTIONS, TASK_CASES, TASK_OPTIONS)
from evals.variant_sweep import (FIN_OPTIONS, FIN_TESTS, HR_OPTIONS,  # noqa: E402
                                 HR_TESTS)

SETS = [
    ("HR", HR_OPTIONS, HR_TESTS),
    ("FIN", FIN_OPTIONS, FIN_TESTS),
    ("IMG", IMG_OPTIONS, [(q, e) for q, e in IMG_CASES if e]),
    ("TASK", TASK_OPTIONS, [(q, e) for q, e in TASK_CASES if e]),
    ("SHOP", SHOP_OPTIONS, [(q, e) for q, e in SHOP_CASES if e]),
]

# ---------------------------------------------------------------------------
# The v0.8.5/v0.8.8 cross-lingual mixup, re-implemented verbatim so it can be
# switched back on. Only the LINEAR probe ever consumed it.
# ---------------------------------------------------------------------------
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


def _detect_lang(text):
    low = text.lower()
    for ch in low:
        if ch in "äöüß":
            return "de"
    for w in re.findall(r"(?u)\b[\w-]+\b", low):
        if w in _GERMAN_SIGNAL_WORDS:
            return "de"
    return "en"


def _cross_lingual_mixup(X, y, texts):
    parts_x, parts_y = [X], [y]
    groups = {}
    for i, lab in enumerate(y):
        groups.setdefault(lab, []).append(i)
    for lab, idxs in groups.items():
        langs = [_detect_lang(texts[i]) for i in idxs]
        if len(set(langs)) <= 1:
            continue
        for a, la in zip(idxs, langs):
            for b, lb in zip(idxs, langs):
                if la != lb:
                    mid = X[a] + X[b]
                    n = float(np.linalg.norm(mid))
                    mid = mid / n if n > 0 else mid
                    parts_x.append(mid.reshape(1, -1))
                    parts_y.append(np.array([lab]))
    return np.vstack(parts_x), np.concatenate(parts_y)


_orig_augment = H._augment_embeddings


def _augment_with_old_mixup(X, y, mixup=True, noise_std=0.01, seed=42):
    """Re-insert the removed cross-lingual mixup before the standard augment.

    NOTE: the old code passed `texts` in; `_augment_embeddings` does not receive
    them. The head calls it with (X, y) only, so the mixup is applied here using
    the head's own flat_texts via the module-level side channel set below.
    """
    texts = _augment_with_old_mixup.current_texts
    X2, y2 = _cross_lingual_mixup(X, y, texts)
    return _orig_augment(X2, y2, mixup=mixup, noise_std=noise_std, seed=seed)


_augment_with_old_mixup.current_texts = []


def run(clf, *, glossary=None, fastpath=None, old_mixup=False):
    """Returns (ok, total, failures) with the given configuration."""
    if old_mixup:
        H._augment_embeddings = _augment_with_old_mixup
    else:
        H._augment_embeddings = _orig_augment
    ok = tot = 0
    fails = []
    for name, opts, tests in SETS:
        e = DecisionEngine(glossary=glossary, sparse_fastpath=fastpath)
        head = Choice(name="h", options=opts, classifier=clf)
        e.add_head(head)
        # feed the mixup the head's own texts
        _augment_with_old_mixup.current_texts = [
            t for exs in opts.values() for t in exs]
        e.compile()
        for t, x in tests:
            got = e.decide(t).h
            tot += 1
            if got == x:
                ok += 1
            else:
                fails.append((name, t, x, got))
    return ok, tot, fails


def main():
    print("=" * 90)
    print("CENTROID vs LINEAR on the 60 frozen cases — per-case attribution")
    print("=" * 90)

    print("\n[A] Baseline reproduction (no glossary, no fast path)")
    c_ok, n, c_f = run("centroid")
    l_ok, _, l_f = run("linear")
    print(f"   centroid {c_ok}/{n}   linear {l_ok}/{n}   -> delta {c_ok - l_ok:+d}")

    print("\n[B] CONTROL: does the new glossary change it? (test config uses none)")
    for label, gl in [("glossary=ON", __import__("klix").manufacturing_glossary()),
                      ("fastpath=ON", None)]:
        kw = {"glossary": gl} if gl else {"fastpath": True}
        c2, _, _ = run("centroid", **kw)
        l2, _, _ = run("linear", **kw)
        print(f"   {label:12} centroid {c2}/{n}   linear {l2}/{n}   delta {c2 - l2:+d}"
              f"   (baseline delta {c_ok - l_ok:+d})")

    print("\n[C] SUSPECT: the removed _cross_lingual_mixup, re-injected")
    c3, _, _ = run("centroid", old_mixup=True)
    l3, _, l3f = run("linear", old_mixup=True)
    print(f"   with old mixup  centroid {c3}/{n}   linear {l3}/{n}   delta {c3 - l3:+d}")
    print(f"   without mixup   centroid {c_ok}/{n}   linear {l_ok}/{n}   delta {c_ok - l_ok:+d}")

    print("\n[D] WHICH cases does each classifier miss? (no glossary, no fastpath)")
    cl = {(s, t) for s, t, _, _ in c_f}
    ll = {(s, t) for s, t, _, _ in l_f}
    only_linear_misses = sorted(cl - ll)
    only_centroid_misses = sorted(ll - cl)
    print(f"   cases ONLY centroid gets wrong ({len(only_linear_misses)}):")
    for s, t in only_linear_misses:
        truth = next((gg for ss, tt, gg, _ in c_f if ss == s and tt == t), "?")
        cent = "wrong" if (s, t) in cl else "ok"
        lin = "wrong" if (s, t) in ll else "ok"
        print(f"     [{s}] {t[:58]!r}")
        print(f"          expected={truth!r}  centroid={cent}  linear={lin}")
    print(f"   cases ONLY linear gets wrong ({len(only_centroid_misses)}):")
    for s, t in only_centroid_misses:
        truth = next((gg for ss, tt, gg, _ in l_f if ss == s and tt == t), "?")
        print(f"     [{s}] {t[:58]!r}   expected={truth!r}")

    print("\n[E] With the old mixup: which cases flip back?")
    ll3 = {(s, t) for s, t, _, _ in l3f}
    gained = sorted(ll - ll3)
    lost = sorted(ll3 - ll)
    print(f"   linear FIXES with old mixup ({len(gained)}): {gained}")
    print(f"   linear BREAKS with old mixup ({len(lost)}): {lost}")


if __name__ == "__main__":
    main()

"""Ablation: does the glossary replace bilingual anchors, or add to them?

The claim under test was made in conversation and never measured: "a glossary
gives you both languages while you author monolingual anchors". Plausible, but the
alternative is simpler — just write the anchors in both languages, which needs no
glossary at all. This runs the four cells on the SAME schema and the SAME test
cases, for BOTH classifiers, with bootstrap confidence intervals:

  (a) EN anchors only,     no glossary
  (b) EN anchors only,     + glossary
  (c) EN + DE anchors,     no glossary
  (d) EN + DE anchors,     + glossary

The decisive questions:

  * Does (b) reach (c)? Then the glossary buys the cross-lingual bridge while you
    author in one language, and further glossary work pays off.
  * Or is (c) clearly better? Then bilingual anchors are the cheaper answer, and
    the value of more glossary work is limited.

Also reported: (a) and (b) against the GERMAN test cases only. That is where a
monolingual-English schema has to be carried by the glossary — if the bridge works
anywhere, it works there, and it is the configuration that makes the whole question
practical.

Honest limits, stated up front
------------------------------
* n = 20 test cases (10 EN + 10 DE) and 20 anchors (4 per class). This is a small
  benchmark; the CI is wide and the per-cell differences are often not resolvable.
  Bootstrap CIs are printed so that is visible instead of implied.
* The anchor sets differ in SIZE between (a)/(b) and (c)/(d) — 4 vs 2 per class is
  forced by the design (that is the comparison), but it means (c)/(d) have both
  twice the anchor volume and two languages. A difference between (b) and (c) is
  therefore not attributable to language coverage alone.
* The glossary is the shipped `curated()` preset. It covers 12 of 20 domain probes
  in this schema (measured separately), so (b) is a partial bridge, not a full one.

Run: uv run python -m evals.glossary_vs_bilingual_anchors
"""
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from klix import Choice, DecisionEngine  # noqa: E402
from klix.glossaries import curated  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Anchors: the 4 bilingual entries of the existing benchmark, split by language.
# EN_ONLY = the 2 English anchors per class; BILINGUAL = all 4 (same texts, so the
# two configurations differ only in language coverage, not in phrasing).
# ---------------------------------------------------------------------------
EN_ONLY = {
    "billing": ["refund for my order is missing", "the invoice amount is wrong"],
    "technical": ["the server keeps crashing", "software update fails with an error"],
    "facility": ["the elevator is stuck between floors", "water leaks from the ceiling"],
    "hr": ["how do i apply for parental leave", "i need a copy of my employment contract"],
    "security": ["suspicious login from another country", "someone tried to hack our admin account"],
}

BILINGUAL = {
    "billing": EN_ONLY["billing"] + ["die rechnung wurde doppelt abgebucht", "gutschrift fehlt auf dem konto"],
    "technical": EN_ONLY["technical"] + ["vpn verbindung bricht staendig ab", "der laptop startet nicht mehr"],
    "facility": EN_ONLY["facility"] + ["die heizung im buero ist kaputt", "parkplatz licht ist ausgefallen"],
    "hr": EN_ONLY["hr"] + ["gehaltsabrechnung stimmt nicht", "urlaub beantragen fuer august"],
    "security": EN_ONLY["security"] + ["ransomware hat den fileserver verschluesselt", "phishing mail an alle mitarbeiter"],
}

TEST_EN = [
    ("you charged my card twice this month", "billing"),
    ("where is the refund for the returned item", "billing"),
    ("my monitor stays black after boot", "technical"),
    ("the wifi never connects on my laptop", "technical"),
    ("the coffee machine in the kitchen leaks", "facility"),
    ("someone broke the glass door handle", "facility"),
    ("i want to extend my parental leave", "hr"),
    ("my payslip shows zero hours", "hr"),
    ("someone logged into my email from abroad", "security"),
    ("all our files are encrypted, ransom demanded", "security"),
]

TEST_DE = [
    ("meine kreditkarte wurde doppelt belastet", "billing"),
    ("wo bleibt die erstattung fuer die stornierte bestellung", "billing"),
    ("der bildschirm bleibt nach dem start schwarz", "technical"),
    ("wlan verbindet sich nicht auf meinem laptop", "technical"),
    ("die kaffeemaschine in der kueche tropft", "facility"),
    ("jemand hat den tuerknauf aus glas zerbrochen", "facility"),
    ("ich moechte meine elternzeit verlaengern", "hr"),
    ("auf meiner abrechnung stehen null stunden", "hr"),
    ("unbekannte haben sich in mein postfach eingeloggt", "security"),
    ("alle unsere dateien sind verschluesselt, loesegeld gefordert", "security"),
]

CELLS = [
    ("(a) EN anchors only, no glossary", EN_ONLY, False),
    ("(b) EN anchors only + glossary", EN_ONLY, True),
    ("(c) EN+DE anchors, no glossary", BILINGUAL, False),
    ("(d) EN+DE anchors + glossary", BILINGUAL, True),
]
CLASSIFIERS = ["centroid", "linear"]

N_BOOT = 2000
SEED = 20260928


def run_cell(anchors, use_glossary, classifier, tests):
    """Returns (correct_flags, latency_ms_median)."""
    gloss = curated() if use_glossary else None
    eng = DecisionEngine(glossary=gloss)
    eng.add_head(Choice(name="c", options=anchors, classifier=classifier))
    eng.compile()
    flags, lat = [], []
    for text, expected in tests:
        res = eng.decide(text)
        flags.append(1 if res.c == expected else 0)
        lat.append(res.latency_ms)
    return flags, statistics.median(lat)


def bootstrap_ci(flags_a, flags_b, n_boot=N_BOOT, seed=SEED):
    """CI of the paired difference (b - a), resampling the test cases."""
    rng = random.Random(seed)
    n = len(flags_a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(flags_b[i] - flags_a[i] for i in idx) / n)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot)]
    point = (sum(flags_b) - sum(flags_a)) / n
    return point, lo, hi


def main():
    tests_all = TEST_EN + TEST_DE
    results = {}

    print("=" * 104)
    print("ABLATION — glossary vs. bilingual anchors (same schema, same test cases)")
    print("=" * 104)
    print(f"  anchors   : (a)/(b) {sum(len(v) for v in EN_ONLY.values())} EN | "
          f"(c)/(d) {sum(len(v) for v in BILINGUAL.values())} EN+DE")
    print(f"  tests     : {len(TEST_EN)} EN + {len(TEST_DE)} DE = {len(tests_all)}")
    print(f"  glossary  : curated() preset — covers 12 of 20 probes in this domain")
    print(f"  bootstrap : {N_BOOT} resamples, seed {SEED}, paired over test cases")
    print()

    for classifier in CLASSIFIERS:
        print("#" * 104)
        print(f"# classifier = {classifier}")
        print("#" * 104)
        print(f"| cell | EN | DE | all | EN% | DE% | all% | median latency |")
        print("|---|---|---|---|---|---|---|---|")
        for name, anchors, use_g in CELLS:
            en_f, _ = run_cell(anchors, use_g, classifier, TEST_EN)
            de_f, _ = run_cell(anchors, use_g, classifier, TEST_DE)
            all_f, lat = run_cell(anchors, use_g, classifier, tests_all)
            results[(classifier, name)] = (en_f, de_f, all_f)
            print(f"| {name} | {sum(en_f)}/{len(en_f)} | {sum(de_f)}/{len(de_f)} | "
                  f"{sum(all_f)}/{len(all_f)} | {sum(en_f)/len(en_f):.0%} | "
                  f"{sum(de_f)/len(de_f):.0%} | {sum(all_f)/len(all_f):.0%} | {lat:.1f} ms |")

        # ---- the decisive comparisons ----
        b_all = results[(classifier, CELLS[1][0])][2]
        c_all = results[(classifier, CELLS[2][0])][2]
        d_all = results[(classifier, CELLS[3][0])][2]
        a_all = results[(classifier, CELLS[0][0])][2]

        print()
        print(f"  DECISIVE: does (b) reach (c)?")
        pt, lo, hi = bootstrap_ci(c_all, b_all)
        print(f"    (b) - (c) = {pt:+.1%}  [95% CI {lo:+.1%}, {hi:+.1%}]  "
              f"{'-> indistinguishable' if lo <= 0 <= hi else '-> differs'}")
        pt, lo, hi = bootstrap_ci(c_all, d_all)
        print(f"    (d) - (c) = {pt:+.1%}  [95% CI {lo:+.1%}, {hi:+.1%}]  "
              f"{'-> indistinguishable' if lo <= 0 <= hi else '-> differs'}  "
              f"(does the glossary add anything ON TOP of bilingual anchors?)")
        pt, lo, hi = bootstrap_ci(a_all, b_all)
        print(f"    (b) - (a) = {pt:+.1%}  [95% CI {lo:+.1%}, {hi:+.1%}]  "
              f"{'-> indistinguishable' if lo <= 0 <= hi else '-> differs'}  "
              f"(glossary value on monolingual-EN anchors)")

        # ---- the practical question: German queries on monolingual-EN anchors ----
        a_de = results[(classifier, CELLS[0][0])][1]
        b_de = results[(classifier, CELLS[1][0])][1]
        c_de = results[(classifier, CELLS[2][0])][1]
        pt, lo, hi = bootstrap_ci(a_de, b_de)
        print()
        print(f"  German queries against EN-only anchors (the bridge's job):")
        print(f"    (a) no glossary : {sum(a_de)}/{len(a_de)}")
        print(f"    (b) + glossary  : {sum(b_de)}/{len(b_de)}   "
              f"delta {pt:+.1%} [95% CI {lo:+.1%}, {hi:+.1%}]  "
              f"{'-> indistinguishable' if lo <= 0 <= hi else '-> differs'}")
        print(f"    (c) EN+DE, none : {sum(c_de)}/{len(c_de)}")
        print()

    # ---- persist for the documents ----
    out = {}
    for (clf, name), (en_f, de_f, all_f) in results.items():
        out[f"{clf}|{name}"] = {
            "en": sum(en_f), "de": sum(de_f), "all": sum(all_f),
            "n_en": len(en_f), "n_de": len(de_f), "n_all": len(all_f),
        }
    (REPO / "evals" / "glossary_vs_anchors_result.json").write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print("wrote evals/glossary_vs_anchors_result.json")


if __name__ == "__main__":
    main()

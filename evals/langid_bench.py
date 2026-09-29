"""Language-ID benchmark: accuracy, abstention, latency, profile pruning.

Two disjoint probe sets, deliberately:

* ``TUNING``   — the operating point (score floor, margin) is chosen on this set.
* ``HOLDOUT``  — never used for tuning; every figure quoted in the README comes
  from here. Tuning and reporting on the same sentences is how a language
  identifier ends up looking better than it is, and short-string LID is exactly
  the regime where that flatters the numbers most.

Both sets are written in the register klix actually sees — short ticket/query
text, 4-12 words — in the ten languages klix ships vocabulary for. They are
hand-written, not machine-translated: a translated probe inherits the source
language's structure, which is precisely the signal being measured.

Run: uv run python -m evals.langid_bench
"""
from __future__ import annotations

import statistics
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from klix import langid  # noqa: E402
from klix.glossaries import language_packs  # noqa: E402

# fmt: off
TUNING: dict[str, list[str]] = {
    "de": ["das foerderband steht seit heute morgen",
           "die taktzeit hat sich nach dem neustart verdoppelt",
           "ersatzteil fuer die hydraulikeinheit fehlt",
           "bitte die wartung fuer naechste woche einplanen",
           "die anmeldung am system funktioniert nicht mehr"],
    "en": ["the conveyor belt stopped this morning",
           "cycle time doubled after the restart",
           "spare part for the hydraulic unit is missing",
           "please schedule maintenance for next week",
           "the login to the system no longer works"],
    "fr": ["le convoyeur est arrete depuis ce matin",
           "le temps de cycle a double apres le redemarrage",
           "la piece de rechange pour l unite hydraulique manque",
           "veuillez planifier la maintenance pour la semaine prochaine",
           "la connexion au systeme ne fonctionne plus"],
    "es": ["la cinta transportadora se ha detenido esta manana",
           "el tiempo de ciclo se ha duplicado tras el reinicio",
           "falta la pieza de repuesto para la unidad hidraulica",
           "por favor planifique el mantenimiento para la proxima semana",
           "el acceso al sistema ya no funciona"],
    "it": ["il nastro trasportatore si e fermato stamattina",
           "il tempo di ciclo e raddoppiato dopo il riavvio",
           "manca il pezzo di ricambio per l unita idraulica",
           "si prega di pianificare la manutenzione per la prossima settimana",
           "l accesso al sistema non funziona piu"],
    "pt": ["a esteira transportadora parou hoje de manha",
           "o tempo de ciclo dobrou apos o reinicio",
           "falta a peca de reposicao para a unidade hidraulica",
           "por favor agende a manutencao para a proxima semana",
           "o acesso ao sistema ja nao funciona"],
    "nl": ["de transportband is vanmorgen gestopt",
           "de cyclustijd is na de herstart verdubbeld",
           "het reserveonderdeel voor de hydraulische eenheid ontbreekt",
           "plan alstublieft het onderhoud voor volgende week",
           "de toegang tot het systeem werkt niet meer"],
    "pl": ["przenosnik tasmowy zatrzymal sie rano",
           "czas cyklu podwoil sie po restarcie",
           "brakuje czesci zamiennej do jednostki hydraulicznej",
           "prosze zaplanowac konserwacje na przyszly tydzien",
           "dostep do systemu juz nie dziala"],
    "sv": ["transportbandet har stannat i morse",
           "cykeltiden har fordubblats efter omstarten",
           "reservdelen till hydraulikenheten saknas",
           "planera underhall for nasta vecka",
           "atkomsten till systemet fungerar inte langre"],
    "da": ["transportbaandet er stoppet i morges",
           "cyklustiden er fordoblet efter genstarten",
           "reservedelen til den hydrauliske enhed mangler",
           "planlaeg venligst vedligeholdelse i naeste uge",
           "adgangen til systemet virker ikke laengere"],
}

HOLDOUT: dict[str, list[str]] = {
    "de": ["stoerung an der anlage melden",
           "der sensor liefert keine messwerte",
           "wann ist der naechste liefertermin",
           "die qualitaet der charge ist schlecht",
           "zugriff auf die datenbank verweigert"],
    "en": ["report an incident on the line",
           "the sensor returns no measurements",
           "when is the next delivery date",
           "the quality of the batch is poor",
           "access to the database denied"],
    "fr": ["signaler une panne sur la ligne",
           "le capteur ne fournit aucune mesure",
           "quelle est la prochaine date de livraison",
           "la qualite du lot est mauvaise",
           "acces a la base de donnees refuse"],
    "es": ["reportar una averia en la linea",
           "el sensor no devuelve mediciones",
           "cuando es la proxima fecha de entrega",
           "la calidad del lote es mala",
           "acceso a la base de datos denegado"],
    "it": ["segnalare un guasto sulla linea",
           "il sensore non fornisce misure",
           "quando e la prossima data di consegna",
           "la qualita del lotto e scarsa",
           "accesso al database negato"],
    "pt": ["reportar uma avaria na linha",
           "o sensor nao devolve medicoes",
           "quando e a proxima data de entrega",
           "a qualidade do lote e ma",
           "acesso a base de dados negado"],
    "nl": ["meld een storing op de lijn",
           "de sensor geeft geen metingen",
           "wanneer is de volgende leverdatum",
           "de kwaliteit van de partij is slecht",
           "toegang tot de database geweigerd"],
    "pl": ["zglos awarie na linii",
           "czujnik nie zwraca pomiarow",
           "kiedy jest nastepny termin dostawy",
           "jakosc partii jest zla",
           "odmowa dostepu do bazy danych"],
    "sv": ["rapportera ett fel pa linjen",
           "sensorn ger inga matningar",
           "nar ar nasta leveransdatum",
           "kvaliteten pa partiet ar dalig",
           "atkomst till databasen nekad"],
    "da": ["rapporter en fejl paa linjen",
           "sensoren giver ingen maalinger",
           "hvornaar er naeste leveringsdato",
           "kvaliteten af partiet er daarlig",
           "adgang til databasen naegtet"],
}
# fmt: on

#: Inputs that must NEVER be forced into a language. Czech, Turkish, Finnish and
#: Hungarian are outside the ten; the rest are inside the ten but too short or
#: too generic to carry the signal.
NEGATIVES: list[str] = [
    "dopravnikovy pas se zastavil dnes rano",          # cs
    "konveyor bant bu sabah durdu",                    # tr
    "kuljetin hihna pysahtyi tana aamuna",             # fi
    "a szallitoszalag ma reggel leallt",               # hu
    "ok",
    "hilfe",
    "error",
    "danke",
]


def _accuracy(model: langid.TrigramModel, corpus: dict[str, list[str]], **kw) -> tuple[int, int, list]:
    hits = 0
    total = 0
    misses = []
    for lang, probes in corpus.items():
        for probe in probes:
            r = langid.detect(probe, model, **kw)
            total += 1
            if r.lang == lang:
                hits += 1
            else:
                misses.append((lang, r.lang, r.reason, probe))
    return hits, total, misses


def _false_confident(model: langid.TrigramModel, **kw) -> list[tuple]:
    out = []
    for probe in NEGATIVES:
        r = langid.detect(probe, model, **kw)
        if r.lang is not None:
            out.append((r.lang, r.confidence, probe))
    return out


def bench_latency(model: langid.TrigramModel, texts: list[str], rounds: int = 400) -> dict:
    for t in texts:
        langid.detect(t, model)
    samples = []
    for _ in range(rounds):
        for t in texts:
            t0 = time.perf_counter()
            langid.detect(t, model)
            samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return {
        "n": len(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": samples[int(0.95 * (len(samples) - 1))],
        "mean_ms": statistics.fmean(samples),
    }


def main() -> None:
    t0 = time.perf_counter()
    model = langid.build_model()
    build_ms = (time.perf_counter() - t0) * 1000

    print("=" * 88)
    print("TRIGRAM LANGUAGE IDENTIFICATION — klix")
    print("=" * 88)
    print(f"languages        : {', '.join(model.langs)}")
    print(f"profiles         : {sum(model.n_grams().values())} {model.n}-grams "
          f"({model.top_n} per language, {model.size_bytes / 1024:.1f} KiB)")
    print(f"model build      : {build_ms:.1f} ms (once, lazy)")
    print()

    # -- threshold calibration, TUNING split only -------------------------
    win_scores, win_margins = [], []
    for lang, probes in TUNING.items():
        for probe in probes:
            sc = model.score(probe)
            ranked = sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))
            if ranked[0][0] == lang:
                win_scores.append(ranked[0][1])
                win_margins.append(ranked[0][1] - ranked[1][1])
    win_scores.sort()
    win_margins.sort()
    p10 = win_scores[int(0.10 * (len(win_scores) - 1))]
    p25m = win_margins[int(0.25 * (len(win_margins) - 1))]
    print("threshold calibration (TUNING split only — never the holdout):")
    print(f"  winner score among correct : p10 {p10:.3f}  min {win_scores[0]:.3f}"
          f"  median {statistics.median(win_scores):.3f}")
    print(f"  margin among correct       : p25 {p25m:.3f}  min {win_margins[0]:.3f}"
          f"  median {statistics.median(win_margins):.3f}")
    print(f"  defaults in use            : floor {langid.DEFAULT_SCORE_FLOOR}"
          f"  margin {langid.DEFAULT_MARGIN_MIN}  (calibrated: floor {p10:.2f},"
          f" margin {p25m:.2f})")
    print()

    tun = _accuracy(model, TUNING)
    hol = _accuracy(model, HOLDOUT)
    print(f"TUNING set (operating point chosen here): {tun[0]}/{tun[1]}"
          f" = {100 * tun[0] / tun[1]:.1f}%")
    print(f"HOLDOUT set (everything reported):        {hol[0]}/{hol[1]}"
          f" = {100 * hol[0] / hol[1]:.1f}%")
    print()
    print("per language (HOLDOUT):")
    for lang in model.langs:
        probes = HOLDOUT[lang]
        got = [langid.detect(p, model) for p in probes]
        hits = sum(1 for g in got if g.lang == lang)
        reasons = {}
        for g in got:
            reasons[g.reason] = reasons.get(g.reason, 0) + 1
        detail = " ".join(f"{k}:{v}" for k, v in sorted(reasons.items()))
        print(f"  {lang}  {hits}/{len(probes)}   {detail}")
    print()
    wrong = [(a, b, r, p) for a, b, r, p in hol[2] if b is not None]
    abst = [(a, b, r, p) for a, b, r, p in hol[2] if b is None]
    print(f"HOLDOUT misses: {len(wrong)} wrong, {len(abst)} abstained")
    for a, b, r, p in wrong:
        print(f"  WRONG  {a} -> {b:3} ({r}) {p!r}")
    for a, b, r, p in abst:
        print(f"  ABSTAIN {a}       ({r}) {p!r}")
    print()

    print("out-of-set / too-short input must not be forced:")
    fc = _false_confident(model)
    if not fc:
        print("  0 of 8 inputs forced into a language (all abstained)")
    for lang_, conf, probe in fc:
        print(f"  FALSE CONFIDENT {lang_} ({conf:.3f}) {probe!r}")
    print()

    print("corpus ablation — which training vocabulary identifies the language:")
    for label, include_content, top_ns in (
        ("content + function words", True, (200, 400)),
        ("function words only", False, (100, 200, 400)),
    ):
        for top_n in top_ns:
            m = langid.build_model(
                packs=language_packs(include_content=include_content),
                top_n=top_n, lang_first=True)
            h = _accuracy(m, HOLDOUT)
            t = _accuracy(m, TUNING)
            print(f"  {label:26} top_n={top_n:4d}  holdout {h[0]:2d}/{h[1]}"
                  f" = {100 * h[0] / h[1]:5.1f}%   tuning {100 * t[0] / t[1]:5.1f}%"
                  f"   {m.size_bytes / 1024:5.1f} KiB")
    print()

    # -- decision quality, not just coverage ------------------------------
    # A "correct" call is only meaningful if most calls are made at all. Report
    # both, on both splits, because the operating point was chosen on TUNING.
    print("decision quality (decided calls only = precision; abstentions are not errors):")
    for split_name, corpus in (("TUNING", TUNING), ("HOLDOUT", HOLDOUT)):
        decided = 0
        correct = 0
        abstain: dict[str, int] = {}
        for lang, probes in corpus.items():
            for probe in probes:
                r = langid.detect(probe, model)
                if r.lang is None:
                    abstain[r.reason] = abstain.get(r.reason, 0) + 1
                else:
                    decided += 1
                    correct += (r.lang == lang)
        prec = (100 * correct / decided) if decided else 0.0
        print(f"  {split_name:8} {correct:2d}/{decided:2d} decided correct"
              f" = {prec:5.1f}% precision, {decided}/{sum(len(v) for v in corpus.values())}"
              f" covered, reasons: {dict(sorted(abstain.items()))}")
    print()

    # -- the word channel, on its own and combined -----------------------
    print("function-word channel ablation (HOLDOUT):")
    for ww, label in ((0.0, "n-gram channel only"), (1.0, "weight 1.0"),
                      (2.0, "weight 2.0"), (4.0, "weight 4.0")):
        decided = correct = 0
        for lang, probes in HOLDOUT.items():
            for probe in probes:
                r = langid.detect(probe, model, word_weight=ww)
                if r.lang is not None:
                    decided += 1
                    correct += (r.lang == lang)
        prec = (100 * correct / decided) if decided else 0.0
        print(f"  {label:22} {correct:2d}/{decided:2d} = {prec:5.1f}% precision,"
              f" {decided}/50 covered")
    print()

    texts = [t for probes in HOLDOUT.values() for t in probes]
    lat = bench_latency(model, texts)
    print(f"latency          : median {lat['median_ms'] * 1000:.1f} us   "
          f"mean {lat['mean_ms'] * 1000:.1f} us   "
          f"p95 {lat['p95_ms'] * 1000:.1f} us   (n={lat['n']})")
    print(f"budget           : 0.1 ms = 100 us  ->  "
          f"{'WITHIN' if lat['p95_ms'] < 0.1 else 'EXCEEDED'} at p95")
    print()

    print("profile pruning (top_n) — accuracy on HOLDOUT vs footprint:")
    for top_n in (50, 100, 200, 400, 800):
        m = langid.build_model(top_n=top_n)
        h = _accuracy(m, HOLDOUT)
        print(f"  top_n={top_n:4d}  {h[0]}/{h[1]} = {100 * h[0] / h[1]:5.1f}%   "
              f"{sum(m.n_grams().values()):5d} grams   {m.size_bytes / 1024:5.1f} KiB")
    print()


if __name__ == "__main__":
    main()

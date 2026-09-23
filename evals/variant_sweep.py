"""Varianten-Messung auf UNVERÄNDERTEN Daten aus allen Testrunden.
Vergleicht max/fixed (Status quo) gegen topk-Pooling, coverage-boost und Kombis."""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from klix import DecisionEngine, Choice, Score  # noqa: E402
from evals.eval_domains import (  # noqa: E402
    IMG_OPTIONS, TASK_OPTIONS, SHOP_OPTIONS,
    IMG_CASES, TASK_CASES, SHOP_CASES,
)

HR_OPTIONS = {
    "urlaub": [
        "urlaub beantragen",
        "please approve my leave days",
        "wie viele urlaubstage habe ich noch offen",
        "i would like to plan my parental leave starting in march and need the forms",
    ],
    "gehalt": [
        "salary payment came in late",
        "die gehaltsabrechnung stimmt nicht",
        "in meiner abrechnung fehlt die sonderzahlung fuer das letzte quartal",
        "my payslip is missing deductions",
    ],
    "vertrag": [
        "ich brauche mein arbeitszeugnis",
        "need a copy of my employment contract",
        "nach der heirat moechte ich meinen nachnamen im arbeitsvertrag aendern lassen",
    ],
    "benefits": [
        "firmenhandy vertrag verlaengern",
        "how do i use the jobticket subsidy",
        "betriebliche altersvorsorge erhoehen",
        "which additional insurances can employees choose in this years benefits catalog",
    ],
}

FIN_OPTIONS = {
    "rechnung": [
        "approve the supplier invoice",
        "rechnung 2201 ist zu pruefen",
        "die neue rechnung muss freigegeben werden",
        "der lieferant schickt dieselbe rechnung zum zweiten mal, bitte mit bestellung abgleichen",
    ],
    "spesen": [
        "taxi rechnung einreichen",
        "submit travel expenses for the trip",
        "meine spesenauszahlung fehlt",
        "fuer die kundenbesuche im september reiche ich fahrt und hotelkosten mit belegen ein",
    ],
    "steuern": [
        "steuernummer im system aendern",
        "vat id number on the invoice is invalid",
        "umsatzsteuererklärung vorbereiten",
    ],
    "budget": [
        "cost center 4210 exceeded its budget",
        "restbudget am jahresende ausgeben",
        "quarterly budget forecast needs review",
    ],
}

HR_TESTS = [
    ("urlaub?", "urlaub"),
    ("extend my parental leave by two months", "urlaub"),
    ("boss approved my week off in august", "urlaub"),
    ("lohn für november zu spät", "gehalt"),
    ("my payslip shows zero hours", "gehalt"),
    ("ich habe meine gehaltsabrechnung für juni nicht erhalten", "gehalt"),
    ("referenzschreiben für neue bewerbung", "vertrag"),
    ("probation period ends next week, sign what?", "vertrag"),
    ("kündigung meines arbeitsvertrags zum oktober", "vertrag"),
    ("deutschticket zuschuss beantragen", "benefits"),
    ("when does the new pension plan start", "benefits"),
    ("ich moechte die betriebliche altersvorsorge aufstocken und brauche das formular", "benefits"),
]

FIN_TESTS = [
    ("spesenabrechnung einreichen", "spesen"),
    ("did my expense refund arrive", "spesen"),
    ("reisekosten für die messe hochladen", "spesen"),
    ("invoice 8871 double paid", "rechnung"),
    ("rechnung 5590 stornieren", "rechnung"),
    ("supplier demands payment within seven days", "rechnung"),
    ("umsatzsteuer melden", "steuern"),
    ("tax audit starts monday, prepare everything", "steuern"),
    ("die steuernummer der filiale hat sich geaendert", "steuern"),
    ("restbudget fuer schulungen nutzen", "budget"),
    ("marketing wants to shift budget into q4 campaigns", "budget"),
    ("forecast shows overspend in it", "budget"),
]

VARIANTS = [
    ("max+fixed (Status quo)", {}),
    ("topk2+fixed", {"label_aggregation": "topk", "label_topk": 2}),
    ("max+coverage", {"keyword_boost_mode": "coverage"}),
    ("topk2+coverage", {"label_aggregation": "topk", "label_topk": 2, "keyword_boost_mode": "coverage"}),
    ("topk3+coverage", {"label_aggregation": "topk", "label_topk": 3, "keyword_boost_mode": "coverage"}),
]


def measure(options: dict, tests: list, variants: list) -> dict:
    out = {}
    for name, kwargs in variants:
        eng = DecisionEngine()
        eng.add_head(Choice(name="h", options=options, **kwargs))
        eng.compile()
        ok = sum(1 for t, e in tests if eng.decide(t).h == e)
        out[name] = f"{ok}/{len(tests)}"
    return out


sets = [
    ("HR (12)", HR_OPTIONS, HR_TESTS),
    ("FIN (12)", FIN_OPTIONS, FIN_TESTS),
    ("IMAGE (12)", IMG_OPTIONS, IMG_CASES[:3] if False else [(q, e) for q, e in IMG_CASES if e]),  # only labeled
    ("TASK (12)", TASK_OPTIONS, [(q, e) for q, e in TASK_CASES if e]),
    ("SHOP (12)", SHOP_OPTIONS, [(q, e) for q, e in SHOP_CASES if e]),
]

print(f"{'VARIANTE':26s} " + " ".join(f"{n:>12s}" for n, _, _ in sets))
print("-" * 92)
results = {}
for name, options, tests in sets:
    results[name] = measure(options, tests, VARIANTS)
for vname, _ in VARIANTS:
    row = f"{vname:26s} "
    for name, _, _ in sets:
        row += f"{results[name][vname]:>12s} "
    print(row)

# Details: welche Tickets ändern sich?
print("\n--- HR-Detailvergleich: Status quo vs. topk2+coverage ---")
eng_old = DecisionEngine()
eng_old.add_head(Choice(name="h", options=HR_OPTIONS))
eng_old.compile()
eng_new = DecisionEngine()
eng_new.add_head(Choice(name="h", options=HR_OPTIONS,
                        label_aggregation="topk", label_topk=2, keyword_boost_mode="coverage"))
eng_new.compile()
for ticket, expected in HR_TESTS:
    old = eng_old.decide(ticket).h
    new = eng_new.decide(ticket).h
    if old != new:
        ok = "FIXED!" if new == expected else "broke" if old == expected else "still wrong"
        print(f"  {ok:11s}: [{old} -> {new}] (expected {expected}) | {ticket}")
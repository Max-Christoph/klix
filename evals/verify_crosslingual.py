"""Verify classifier="auto" and translate_fn cross-lingual augmentation actually
help the DE weakness, using the bilingual benchmark dataset."""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from klix import DecisionEngine, Choice  # noqa: E402
from evals.benchmark_bilingual import OPTIONS, TEST_EN, TEST_DE  # noqa: E402


def run(classifier, translate_fn=None, **kw):
    eng = DecisionEngine()
    eng.add_head(Choice(name="c", options=OPTIONS, classifier=classifier,
                        translate_fn=translate_fn, **kw))
    eng.compile()
    eff = getattr(eng.heads[0], "_effective_classifier", "?")
    en_ok = sum(1 for t, e in TEST_EN if eng.decide(t).c == e)
    de_ok = sum(1 for t, e in TEST_DE if eng.decide(t).c == e)
    return en_ok, de_ok, eff


# 1. baseline: nearest / linear
for c in ["nearest", "linear", "auto"]:
    en, de, eff = run(c)
    print(f"{c:10s} -> eff={eff:8s} EN={en}/10 DE={de}/10  (combined {en+de}/20)")

# 2. auto with a fake translator (dictionary-based, to prove the hook works)
DICT = {
    "refund for my order is missing": "erste erstatung fuer meine bestellung fehlt",
    "the invoice amount is wrong": "der rechnungsbetrag ist falsch",
    "the server keeps crashing": "der server stuerzt staendig ab",
    "software update fails with an error": "software update schlaegt fehl",
    "the elevator is stuck between floors": "der aufzug steckt fest",
    "water leaks from the ceiling": "wasser tropft von der decke",
    "how do i apply for parental leave": "wie beantrage ich elternzeit",
    "i need a copy of my employment contract": "ich brauche eine kopie meines arbeitsvertrags",
    "suspicious login from another country": "verdaechtiger login aus fremdem land",
    "someone tried to hack our admin account": "jemand versuchte unser admin konto zu hacken",
}
DICT.update({
    "die rechnung wurde doppelt abgebucht": "the invoice was charged twice",
    "gutschrift fehlt auf dem konto": "credit note is missing on the account",
    "vpn verbindung bricht staendig ab": "vpn connection keeps dropping",
    "der laptop startet nicht mehr": "the laptop won't boot anymore",
    "die heizung im buero ist kaputt": "the office heating is broken",
    "parkplatz licht ist ausgefallen": "parking lot light is out",
    "gehaltsabrechnung stimmt nicht": "payslip is wrong",
    "urlaub beantragen fuer august": "apply for leave in august",
    "ransomware hat den fileserver verschluesselt": "ransomware encrypted the file server",
    "phishing mail an alle mitarbeiter": "phishing email to all employees",
})

def fake_translate(text, target):
    # only translate if we know the word; else return None (best-effort)
    return DICT.get(text.lower())

en, de, eff = run("linear", translate_fn=fake_translate)
print(f"\nlinear + translate_fn -> eff=linear EN={en}/10 DE={de}/10 (combined {en+de}/20)")

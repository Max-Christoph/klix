"""Test: translate_fn-Fehler werden gemeldet statt stumm geschluckt.

Regression-Test fuer die Review-Aenderung. Zwei Dinge werden geprueft:

  1. Ein kaputter translate_fn darf compile() NICHT brechen, muss aber
     einmal warnen (vorher verschwand der Fehler lautlos).
  2. translate_fn wirkt NUR auf dem linear/hybrid-Pfad. Auf dem
     nearest-Pfad (Default) wird der Hook gar nicht aufgerufen -- das ist
     Absicht (dort gibt es keine Trainingsmatrix), aber es war
     undokumentiert. Der Test pinnt dieses Verhalten fest.
"""
import warnings

import pytest

from klix import Choice, DecisionEngine


OPT = {
    "deutsch": ["der server ist ausgefallen", "kein netzwerkzugang im buero"],
    "englisch": ["the conveyor belt stopped", "the printer is offline"],
}


def _broken_translate(text, target):
    """Immer kaputt -- simuliert einen Tippfehler in der Nutzerfunktion."""
    raise RuntimeError("translator backend unreachable")


def _working_translate(text, target):
    tabelle = {
        ("der server ist ausgefallen", "en"): "the server is down",
        ("kein netzwerkzugang im buero", "en"): "no network access in the office",
    }
    return tabelle.get((text.lower(), target))


# ---------------------------------------------------------------------------
# 1. Warnung statt stillem Schlucken (auf dem linear-Pfad, wo der Hook laeuft)
# ---------------------------------------------------------------------------

def test_broken_translate_warns_but_does_not_raise():
    """compile() darf nicht brechen, muss aber warnen."""
    eng = DecisionEngine()
    eng.add_head(Choice(name="k", options=OPT, classifier="linear",
                        translate_fn=_broken_translate))

    with pytest.warns(UserWarning, match="translate_fn"):
        eng.compile()

    res = eng.decide("der server ist ausgefallen")
    assert res.k is not None


def test_warning_names_the_error_and_the_count():
    eng = DecisionEngine()
    eng.add_head(Choice(name="k", options=OPT, classifier="linear",
                        translate_fn=_broken_translate))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eng.compile()

    msgs = [str(w.message) for w in caught if "translate_fn" in str(w.message)]
    assert len(msgs) == 1, f"erwartet genau 1 Warnung, bekam {len(msgs)}"
    assert "RuntimeError" in msgs[0]
    assert "of 4" in msgs[0]


def test_working_translate_does_not_warn():
    eng = DecisionEngine()
    eng.add_head(Choice(name="k", options=OPT, classifier="linear",
                        translate_fn=_working_translate))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eng.compile()

    assert [w for w in caught if "translate_fn" in str(w.message)] == []


def test_translate_none_return_is_not_an_error():
    """Eine Funktion, die None liefert (kein Eintrag), ist kein Fehler."""
    eng = DecisionEngine()
    eng.add_head(Choice(name="k", options=OPT, classifier="linear",
                        translate_fn=lambda t, l: None))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eng.compile()

    assert [w for w in caught if "translate_fn" in str(w.message)] == []


# ---------------------------------------------------------------------------
# 2. translate_fn wirkt nur auf linear/hybrid -- Verhalten festgenagelt
# ---------------------------------------------------------------------------

def test_translate_fn_is_ignored_on_nearest_path():
    """Auf dem nearest-Pfad wird der Hook nicht aufgerufen (dokumentiert).

    Kein Fehler -- dort existiert keine Trainingsmatrix, in die die
    Spiegelung einfliessen koennte. Der Test haelt fest, dass (a) nichts
    bricht und (b) keine Warnung erscheint, weil der Hook nie lief.
    """
    aufrufe = []

    def zaehlender_translate(text, target):
        aufrufe.append((text, target))
        return None

    eng = DecisionEngine()
    eng.add_head(Choice(name="k", options=OPT, classifier="nearest",
                        translate_fn=zaehlender_translate))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eng.compile()

    assert aufrufe == [], "translate_fn darf auf dem nearest-Pfad nicht laufen"
    assert [w for w in caught if "translate_fn" in str(w.message)] == []
    # Engine funktioniert normal.
    assert eng.decide("kein netzwerkzugang im buero").k == "deutsch"


def test_translate_fn_runs_on_linear_path():
    """Gegenprobe: auf dem linear-Pfad wird der Hook tatsaechlich aufgerufen."""
    aufrufe = []

    def zaehlender_translate(text, target):
        aufrufe.append((text, target))
        return None

    eng = DecisionEngine()
    eng.add_head(Choice(name="k", options=OPT, classifier="linear",
                        translate_fn=zaehlender_translate))
    eng.compile()

    assert len(aufrufe) > 0, "translate_fn muss auf dem linear-Pfad laufen"
    # Jeder Aufruf spiegelt in die jeweils andere Sprache.
    for text, target in aufrufe:
        assert target in ("de", "en")


def test_no_translate_fn_no_warning():
    eng = DecisionEngine()
    eng.add_head(Choice(name="k", options=OPT, classifier="linear"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eng.compile()

    assert [w for w in caught if "translate_fn" in str(w.message)] == []

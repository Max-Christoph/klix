"""klix-engine 0.6.0 - Kompletter Überblick in einer Datei.

Ausführen:
    pip install klix-engine   (bzw. uv pip install klix-engine)
    python klix_demo.py

Das Skript zeigt alle Köpfe (Choice, Score, Flag), alle wichtigen Parameter
und die typischen Nutzungsmuster -- direkt lauffähig, vollständig kommentiert.

Neu in 0.4.0 (Abschnitte 8-10):
  - Regeln (force/boost) als harte Signale über der Semantik
  - res.explain(): welche Wörter/Anchors haben entschieden?
  - engine.calibrate(): Schwellenwerte aus Beispielen lernen statt raten
Neu in 0.5.0:
  - erweiterte Stoppwörter (Füllwörter wie "den/hat" überschreiben nichts mehr)
  - decide_batch(): Bulk-Verarbeitung mit einem Embedding-Pass
  - CV-Kalibrierung (k-fold, median über Folds)
Neu in 0.6.0 (Abschnitt 11):
  - engine.validate_anchors(): Klassen-Overlap-Report mit Schärfungsvorschlägen
"""

import time

from klix import BaseHead, Choice, DecisionEngine, Flag, Score
from klix.rules import Rule

# ============================================================================
# 1. ENGINE ANLEGEN
# ============================================================================
# DecisionEngine(model_name=..., stop_words=...)
#   model_name : jedes FastEmbed-kompatible Embedding-Modell
#                (Default: multilingual MiniLM, ~120 MB, einmaliger Download)
#   stop_words : None = eingebaute EN+DE-Liste (Default)
#                []    = Stopwort-Filterung komplett AUS
#                ["die","the",...] = eigene Liste (z. B. dritte Sprache)
engine = DecisionEngine()
# engine = DecisionEngine(stop_words=[])                  # Filter aus
# engine = DecisionEngine(stop_words=["der","die","the"]) # eigene Liste

# ============================================================================
# 2. KÖPFE DEFINIEREN - das ist das gesamte "Training"
# ============================================================================
# Anchors = Beispiel-Sätze pro Klasse. Mehr/bessere Anchors = bessere Qualität.
# Sprachen können gemischt sein (DE/EN), das Modell ist multilingual.

engine.add_head(
    Choice(
        name="route",
        options={
            # Rechnung / Zahlung - deutsche und englische Anchors gemischt
            "billing": [
                "die rechnung wurde doppelt abgebucht",
                "gutschrift fehlt auf dem konto",
                "refund for my order is missing",
                "the invoice amount is wrong",
            ],
            # Technische Störung
            "technical": [
                "der laptop startet nicht mehr",
                "vpn verbindung bricht staendig ab",
                "the server keeps crashing",
                "software update fails with an error",
            ],
            # Gebäude / Infrastruktur
            "facility": [
                "die heizung im buero ist kaputt",
                "wasser tropft von der decke",
                "the elevator is stuck between floors",
                "parkplatz licht ist ausgefallen",
            ],
            # Security-Vorfall
            "security": [
                "verdächtiger login aus fremdem land",
                "ransomware hat den fileserver verschlüsselt",
                "suspicious login from another country",
                "phishing mail an alle mitarbeiter",
            ],
            # Explizite Auffang-Klasse: alles, was in keine Fach-Queue gehört.
            # Vorteil gegenüber nur value=None: solche Tickets bekommen einen
            # echten Bucket und können automatisch geschlossen werden.
            "not_relevant": [
                "kaffee ist alle, bitte neuen besorgen",
                "die kaffeemaschine hat nicht genug pulver für alle mitarbeiter",
                "danke für die tolle hilfe an das team",
                "freundliche grüße und ein schönes wochenende",
                "wer hat den letzten joghurt aus dem kühlschrank genommen",
                "die räume müssten mal wieder geputzt werden",
                "happy birthday to a colleague",
                "just chatting, have a nice weekend everyone",
            ],
        },
        # ---- Wahl des Klassifikators --------------------------------------
        # "nearest" (Default): robuster Anker-Vergleich, gut bei wenigen /
        #     gemischtsprachigen Anchors.
        # "linear": lernt eine Entscheidungsgrenze (LogReg auf den Anchors),
        #     genauer bei einsprachigen Schemata mit 3+ Anchors/Klasse.
        # "auto": wählt selbst - nearest bei gemischten Sprachen, linear bei
        #     einsprachig. Empfohlen, wenn du es nicht manuell entscheiden willst.
        classifier="auto",
        #
        # ---- Optional: Cross-lingual-Brücke -------------------------------
        # translate_fn spiegelt jeden Anchor in die fehlende Sprache.
        # Signatur: fn(text, ziel_sprache) -> übersetzter_text
        # Für Demo hier ein Mini-Wörterbuch; in echt: kleines lokales
        # Übersetzungsmodell (z. B. OPUS-MT) oder ein LLM.
        translate_fn=lambda text, target: {
            ("die rechnung wurde doppelt abgebucht", "en"): "the invoice was charged twice",
            ("gutschrift fehlt auf dem konto", "en"): "credit note is missing on my account",
            ("the invoice amount is wrong", "de"): "der rechnungsbetrag ist falsch",
            ("refund for my order is missing", "de"): "erste erstattung meiner bestellung fehlt",
            ("der laptop startet nicht mehr", "en"): "the laptop won't boot anymore",
            ("vpn verbindung bricht staendig ab", "en"): "the vpn connection keeps dropping",
            ("software update fails with an error", "de"): "software update schlägt mit fehler fehl",
            ("the server keeps crashing", "de"): "der server stürzt ständig ab",
            ("die heizung im buero ist kaputt", "en"): "the office heating is broken",
            ("wasser tropft von der decke", "en"): "water leaks from the ceiling",
            ("the elevator is stuck between floors", "de"): "der aufzug steckt zwischen etagen fest",
            ("parkplatz licht ist ausgefallen", "en"): "the parking lot light is out",
            ("verdächtiger login aus fremdem land", "en"): "suspicious login from a foreign country",
            ("ransomware hat den fileserver verschlüsselt", "en"): "ransomware encrypted the file server",
            ("suspicious login from another country", "de"): "verdächtiger login aus einem anderen land",
            ("phishing mail an alle mitarbeiter", "en"): "phishing email to all employees",
        }.get((text.lower(), target)),
        #
        # ---- Reject-Pol: Texte, die KEINE Klasse verdienen -----------------
        # Bei denen wird das Ticket nicht geroutet, sondern aufgeräumt:
        reject_anchors=[
            "danke für die tolle hilfe",
            "dickes lob an das support team",
            "freundliche grüße und schönes wochenende",
            "just saying hello and thanks",
        ],
        # ---- Score-Floor: zu niedriger score -> ebenfalls value=None ------
        # (zweite Sicherheitsschicht für Off-Domain-Texte ohne Reject-Pol)
        reject_threshold=0.35,
        #
        # ---- Regeln: harte Signale über der Semantik (NEU in 0.4.0) --------
        # mode="force": wenn der Ausdruck matcht, gewinnt das Label SOFORT -
        #     auch gegen die Semantik, den Reject-Pol und reject_threshold.
        #     Das expliziteste Nutzer-Signal ("diese Wörter sind immer X").
        # mode="boost": addiert `weight` auf das Label - kann knappe
        #     Entscheidungen kippen, ohne sie zu erzwingen.
        # Getriggerte Regeln stehen später in details("route")["matched_rules"].
        rules=[
            # Kernserver-IPs und Asset-Tags sind IMMER ein OT-Problem,
            # egal wie der Rest des Satzes klingt:
            Rule(
                label="security",
                pattern=r"(?i)\b(?:ransomware|phishing|hacker)\b",
                mode="force",
                name="security_keywords",
            ),
            # Asset-IDs (plc-xx) sind immer technisch:
            Rule(
                label="technical",
                pattern=r"\bplc-\d+\b",
                mode="boost",
                weight=1.5,
                name="plc_asset_tag",
            ),
        ],
        #
        # ---- Weitere nützliche Knobs ---------------------------------------
        # keyword_boost=0.5          : Gewicht exakter Worttreffer (Asset-IDs wie "plc-34")
        # label_aggregation="topk"   : mittelt die besten 2 Anchors pro Klasse
        #                              (statt Max; sinnvoll ab 4+ Anchors/Klasse)
        # classifier_C=10.0          : Regularisierung der linearen Sonde
    )
)

engine.add_head(
    Score(
        name="urgency",
        # Score = kontinuierliche Achse zwischen zwei Polen.
        # Anchors unten = ruhig, Anchors oben = dringend.
        low_anchors=[
            "kann bis nächste woche warten",
            "keine eile, normale anfrage",
            "just a routine question, no rush",
        ],
        high_anchors=[
            "produktion steht komplett still",
            "notfall, sofort hilfe nötig",
            "critical outage happening right now",
        ],
        min_val=0.0,       # unteres Ende der Skala
        max_val=3.0,       # oberes Ende der Skala
        aggregation="topk",  # robust gegen einzelnen verrauschten Anchor
        sharpness=8.0,     # Steilheit der Sigmoid-Kurve (default 8.0)
    )
)

engine.add_head(
    Flag(
        name="is_security",
        # Flag = Ja/Nein-Entscheidung mit Vertrauenssignalen.
        true_anchors=[
            "hackerangriff auf unser system",
            "ransomware verschlüsselt alle dateien",
            "root login wurde kompromittiert",
        ],
        false_anchors=[
            "der drucker ist papierstau",
            "hardware defekt, bitte tauschen",
            "normales it-problem, nichts besonderes",
        ],
        # Optionaler dritter Pol: wenn "neutral" gewinnt -> value=None
        # (statt geratenem True/False bei Off-Topic-Texten).
        neutral_anchors=["routine anfrage", "allgemeine frage", "sonstiges thema"],
        threshold=0.5,  # Schwelle für True (default 0.5)
        temp=0.12,      # Softmax-Temperatur; niedrig = schärfer (default 0.12)
        # aggregation="topk"  # ab 3+ Anchors pro Pol empfohlen
    )
)

# ============================================================================
# 3. KOMPILIEREN (einmalig, ~50-100 ms)
# ============================================================================
# Lädt Anchors, baut TF-IDF-Indizes, augmentiert und trainiert ggf. die Sonde.
# Ohne expliziten Aufruf passiert das automatisch beim ersten decide().
engine.compile()

# ============================================================================
# 4. ENTSCHEIDUNGEN TREFFEN
# ============================================================================
# Wichtig: der Text wird EINMAL durch das Embedding-Modell geschickt (~10-40 ms),
# alle 3 Köpfe werten dann dieselben Vektoren aus (< 1 ms zusammen).

tickets = [
    "meine karte wurde doppelt belastet, bitte sofort prüfen!",   # -> billing
    "the server is down, production line stopped!",               # -> technical
    "die heizung im seminarraum geht nicht",                      # -> facility
    "jemand hat sich in den admin account eingeloggt, um 3 uhr",  # -> security
    "Ticket-4711: die kaffeemaschine hat nicht mehr genügend Pulver "
    "um unsere ganzen Mitarbeiter zu sättigen! Bitte schnell neues organisieren!",
    # -> not_relevant (Kaffeemaschine/Beschaffung ist keine der Fach-Queues)
    "thanks a lot for the great support, have a nice weekend!",   # -> not_relevant
]

print("=" * 78)
print("klix-engine Demo - 3 Köpfe (Choice + Score + Flag), geteilter Backbone")
print("=" * 78)

for ticket in tickets:
    res = engine.decide(ticket)

    # --- Zugriff auf Ergebnisse ------------------------------------------
    # 1) Als Attribut (nur der 'value'-Teil):
    #       res.route / res.urgency / res.is_security
    # 2) Als volles Dict (alle Details):
    #       res.details("route") / res.details("urgency") / ...
    d_route = res.details("route")
    d_urg = res.details("urgency")
    d_flag = res.details("is_security")

    # ------------------------------------------------------------------
    # PRODUKTIONSMUSTER: zweistufige Entscheidung
    # ------------------------------------------------------------------
    # Eine Auffang-Klasse ("not_relevant") ist bequem, aber gefährlich, wenn
    # sie mit niedriger Konfidenz gewinnt (dann verliert evtl. ein echter
    # Security-Fall dagegen). Deshalb:
    #   1. Security-Flag als VETO: echten Vorfall nie automatisch schließen
    #   2. not_relevant nur bei HOHER Konfidenz automatisch behandeln
    #   3. alles andere -> manuelle Prüfung
    CONF_GATE = 0.6  # Konfidenz-Schwelle für Automatisierung

    if d_route["value"] == "not_relevant" and res.is_security is True:
        action = "MANUELL PRÜFEN (Security-Flag widerspricht Auffang-Klasse!)"
    elif d_route["value"] != "not_relevant" and d_route["confidence"] >= CONF_GATE:
        action = f"-> automatisch in Fach-Queue '{res.route}' legen"
    elif d_route["value"] == "not_relevant" and d_route["confidence"] >= CONF_GATE:
        action = "-> automatisch schließen (nicht fachlich relevant)"
    else:
        action = "-> MANUELLE PRÜFUNG (Konfidenz zu niedrig für Automatik)"

    print(f"\nTicket : {ticket}")
    print(f"  {res!r}")
    print(f"  route       : {res.route!r:12s}  (score={d_route['score']:.2f}, "
          f"conf={d_route['confidence']:.2f}, reject={d_route.get('reject_score', 0):.2f})")
    print(f"  urgency     : {res.urgency:.2f}/3.0  (coverage={d_urg['coverage']:.2f})")
    print(f"  is_security : {res.is_security}  (p={d_flag['probability']:.2f}, "
          f"margin={d_flag['margin']:.2f}, coverage={d_flag['coverage']:.2f})")
    print(f"  >>> Aktion  : {action}")

# ============================================================================
# 5. INTERPRETATION DER SIGNALE  (das Herzstück für produktive Nutzung)
# ============================================================================
print("\n" + "=" * 78)
print("Vertrauenssignale kurz erklärt")
print("=" * 78)
print("""
route  (Choice):
  score    - Ähnlichkeit zum besten Anchor (hybrid aus dense + keyword)
  confidence - Abstand zum Zweitplatzierten; niedrig = grenzwertiger Fall
  reject_score - wie stark ein Reject-Anchor (Smalltalk etc.) matcht;
                 wenn er gewinnt -> value=None
  Alle drei zusammen: value nur vertrauen, wenn score hoch UND
  confidence nicht nahe 0 UND value nicht None.

urgency (Score):
  value    - Projektion auf die Low/High-Achse [min_val, max_val]
  coverage - Ähnlichkeit zum besser passenden Pol.
             < ~0.3 = der Text hat keinen Pol getroffen -> Score ist Rauschen,
             nicht verwenden! Das ist das ehrliche "weiß ich nicht"-Signal.

is_security (Flag):
  probability - Softmax-Vertrauen (KEINE kalibrierte Wahrscheinlichkeit!)
  margin      - |p_true - p_false|; groß = klare Entscheidung
  coverage    - wie stark der Text überhaupt an einen Pol passt
""")

# ============================================================================
# 6. EIGENE KÖPFE  (BaseHead erben)
# ============================================================================
print("=" * 78)
print("Eigener Kopf: RegExExtraction (Ticketsnummer extrahieren)")
print("=" * 78)

import re


class TicketIdHead(BaseHead):
    """Extrahiert IDs wie 'INC-4711' oder 'TICKET-42' aus dem Text.

    Gibt die ID MIT ihrem Original-Präfix zurück — ein Extraktions-Kopf darf
    Daten nicht stillschweigend umschreiben (Transformations-Regel).
    """

    def __init__(self, name: str = "ticket_id"):
        super().__init__(name)
        self.pattern = re.compile(r"(?i)\b(ticket|inc|case)[-# ]?(\d{2,6})\b")

    def get_reference_texts(self) -> list[str]:
        return []  # keine Referenztexte nötig

    def fit(self, backbone) -> None:
        pass  # nichts vorzuberechnen

    def evaluate(self, encoded) -> dict:
        m = self.pattern.search(encoded.text)
        if not m:
            return {"value": None}
        prefix, number = m.group(1).upper(), m.group(2)
        return {"value": f"{prefix}-{number}"}  # original prefix, no rewriting


engine.add_head(TicketIdHead())
engine.compile()  # neuer Kopf -> neu kompilieren

demo = "Ticket-4711: die kaffeemaschine hat nicht mehr genügend Pulver um unsere ganzen Mitarbeiter zu sättigen! Bitte schnell neues organisieren!"
res = engine.decide(demo)
print(f"\nText  : {demo}")
print(f"route : {res.route} | ticket_id: {res.ticket_id} | urgency: {res.urgency}")

# ============================================================================
# 7. REGELN IN AKTION - force und boost (NEU in 0.4.0)
# ============================================================================
print("\n" + "=" * 78)
print("Regeln (force/boost) - harte Signale über der Semantik")
print("=" * 78)

rule_tickets = [
    # Force-Regel: "ransomware" erzwingt security, egal was die Semantik sagt
    ("phishing mail aussieht wie eine rechnung, bitte prüfen", "security erzwungen"),
    # Boost-Regel: "plc-42" hebt technical an (Asset-ID-Signal)
    ("plc-42 sporadische fehler im schichtbetrieb", "technical per boost"),
    # Ohne Regel-Treffer: normale Semantik
    ("die heizung im besprechungsraum geht nicht", "normal geroutet"),
]
for text, expect in rule_tickets:
    res = engine.decide(text)
    d = res.details("route")
    matched = d.get("matched_rules", [])
    forced = d.get("forced_by")
    print(f"\nText : {text}")
    print(f"  Route : {d['value']!r} (score={d['score']:.2f}, conf={d['confidence']:.2f})")
    if forced:
        print(f"  >>> ERZWUNGEN durch Regel {forced!r}")
    elif matched:
        print(f"  >>> Regel getriggert: {matched}")
    else:
        print(f"  >>> keine Regel gegriffen (rein semantisch)")

# ============================================================================
# 8. EXPLAIN - WARUM wurde so entschieden? (NEU in 0.4.0)
# ============================================================================
print("\n" + "=" * 78)
print("res.explain() - Decision Attribution (Debugging)")
print("=" * 78)

explain_tickets = [
    "die rechnung wurde doppelt abgebucht, bitte sofort prüfen",  # klarer billing-Fall
    "jemand hat sich in den admin account eingeloggt, um 3 uhr",  # security
]
for text in explain_tickets:
    res = engine.decide(text)
    exp = res.explain("route")
    print(f"\nText : {text}")
    print(f"  Entscheidung: {exp['value']}")
    for b in exp.get("because", []):
        if b["kind"] == "semantic":
            print(f"    semantisch : Anchor {b['anchor']!r} -> {b['similarity']:.0%}")
        elif b["kind"] == "keyword":
            print(f"    keyword    : {b['token']!r} (Beitrag {b['weight']})")
        elif b["kind"] == "rule":
            print(f"    regel      : {b['rule']}")
    if exp.get("runner_up"):
        print(f"    runner-up  : {exp['runner_up']['label']} ({exp['runner_up']['score']:.2f})")

# Auch Flag und Score erklären sich selbst:
print()
exp_flag = engine.decide("ransomware verschlüsselt unsere daten!").explain("is_security")
print(f"Flag-Interpretation: {exp_flag['interpretation']}")
exp_score = engine.decide("produktion steht, sofort hilfe!").explain("urgency")
print(f"Score-Interpretation: {exp_score['interpretation']}")

# ============================================================================
# 9. CALIBRATE - Schwellenwerte lernen statt raten (NEU in 0.4.0)
# ============================================================================
print("\n" + "=" * 78)
print("engine.calibrate() - Threshold aus Beispielen lernen")
print("=" * 78)

flag_head = next(h for h in engine.heads if h.name == "is_security")
print(f"\nFlag-Threshold vorher: {flag_head.threshold}")
kalib_samples = [
    ("ransomware hat den server verschlüsselt", True),
    ("fremder login im admin konto um 3 uhr", True),
    ("phishing mail an die buchhaltung", True),
    ("unbekannter datenabfluss gestern nacht", True),
    ("der drucker hat papierstau", False),
    ("der monitor flackert", False),
    ("das wlan ist langsam", False),
    ("maus kabel ist kaputt", False),
]
report = engine.calibrate("is_security", kalib_samples)
print(f"Flag-Threshold nachher: {flag_head.threshold} "
      f"(metric={report['metric']}, wert={report['value']:.2f}, n={report['n']})")
if report.get("warning"):
    print(f"  Hinweis: {report['warning']}")

# Score-Kalibrierung: Sharpness + affine Remap aus Zielwerten lernen
print("\nScore-Kalibrierung (Zielwerte statt Standard-Skala):")
score_samples = [
    ("produktion steht komplett still", 3.0),
    ("notfall, alles fällt aus", 3.0),
    ("kritischer ausfall läuft gerade", 2.7),
    ("routinefrage, keine eile", 0.2),
    ("kann bis nächste woche warten", 0.1),
    ("normale anfrage ohne priorität", 0.3),
]
report = engine.calibrate("urgency", score_samples)
print(f"  sharpness={report['sharpness']}, remap: wert ≈ {report['a']:.2f} + {report['b']:.2f}·raw")
check = engine.decide("produktion steht komplett still, alles down!")
print(f"  Kontrolle 'produktion steht komplett still, sofort': urgency={check.urgency} (Ziel ~3.0)")
check2 = engine.decide("routinefrage, kann warten")
print(f"  Kontrolle 'routinefrage, kann warten': urgency={check2.urgency} (Ziel ~0.2)")

# ============================================================================
# 10. BATCH - Massenverarbeitung (NEU in 0.5.0)
# ============================================================================
print("\n" + "=" * 78)
print("engine.decide_batch() - ein Embedding-Pass für alle Texte")
print("=" * 78)

batch_texts = [
    "meine karte wurde doppelt belastet",
    "the server is down",
    "die heizung geht nicht",
    "phishing mail erhalten",
    "kaffee ist alle",
]
t0 = time.perf_counter()
batch = engine.decide_batch(batch_texts)
batch_ms = (time.perf_counter() - t0) * 1000
print(f"\n{len(batch_texts)} Texte in einem Embedding-Pass: {batch_ms:.1f} ms gesamt "
      f"({batch_ms/len(batch_texts):.2f} ms/Item)")
for text, res in zip(batch_texts, batch):
    print(f"  {res.route!r:15s} <- {text}")

# ============================================================================
# 11. VALIDATE_ANCHORS - Klassen-Overlap-Report (NEU in 0.6.0)
# ============================================================================
print("\n" + "=" * 78)
print("engine.validate_anchors_report() - Anchor-Qualität prüfen")
print("=" * 78)
print("\nRead-only Diagnose: überlappende Klassen, Verwechsler-Wörter,")
print("fehlplazierte und doppelte Anchors. Mutiert NIEMALS Anchors.\n")

# Zweite Engine mit absichtlich überlappenden Klassen zum Zeigen des Reports:
overlap_engine = DecisionEngine()
overlap_engine.add_head(
    Choice(
        name="bereich",
        options={
            "billing": [
                "bitte rechnung freigeben",
                "die rechnung muss geprüft werden",
                "rechnung nr 42 freigeben",
            ],
            "finance": [
                "bitte rechnung freigeben",
                "rechnung für die buchhaltung freigeben",
                "rechnungsfreigabe für das finance team",
            ],
            "technical": [
                "server ausgefallen",
                "vpn bricht ab",
                "laptop startet nicht",
            ],
        },
    )
)
overlap_engine.compile()
print(overlap_engine.validate_anchors_report())
print("\n(Legende: !! = hohe Überlappung, ? = mittlere; 'sharpen' = exklusive")
print(" Begriffe, mit denen sich die jeweilige Klasse schärfen lässt)")

# Und für die HAUPTEngine zeigen, dass sie sauber ist:
main_report = engine.validate_anchors_report()
print("\nHaupt-Engine:", main_report.splitlines()[0] if main_report else "keine Findings")

# ============================================================================
# 12. LATEZ-PROFIL
# ============================================================================
print("\n" + "=" * 78)
print("Latenzprofil (50 Durchläufe, median)")
print("=" * 78)
engine.decide("warmup")
times = []
for _ in range(50):
    t0 = time.perf_counter()
    engine.decide("plc-34 meldet fehler, förderband steht sofort")
    times.append((time.perf_counter() - t0) * 1000)
print(f"  3 Köpfe + TicketIdHead: {sorted(times)[25]:.1f} ms median "
      f"(davon ~10-40 ms Embedding-Pass, Kopf-Mathe < 1 ms)")
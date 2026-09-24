"""klix-engine 0.8.1 - the complete production pattern in a single file.

Run:
    pip install klix-engine      (or: uv pip install klix-engine)
    python examples/production_pattern.py

The script walks through every head (Choice, Score, Flag), the important
parameters, and the typical usage patterns -- runnable as-is, fully commented.

Note on language: the anchor texts and ticket examples below are intentionally
mixed German/English. That is the point of the demo -- the backbone model is
multilingual, so anchors in any language work, and the `translate_fn` hook
shows how to mirror anchors into the missing language. All comments and output
labels are English.

New in 0.4.0:
  - Rules (force/boost), res.explain(), engine.calibrate()
New in 0.5.0:
  - extended stop words, decide_batch(), CV calibration
New in 0.6.0:
  - engine.validate_anchors(): class-overlap report
New in 0.7.0 (enterprise hardening):
  - Score min_coverage gate: noisy scores return value=None (+raw_value)
  - calibration warns loudly (UserWarning) for n < 20
  - politeness fillers (bitte/please/danke/thanks) filtered
New in 0.8.0:
  - head gating (suppress_when), DriftMonitor, schema_hash()
"""

import time

from klix import BaseHead, Choice, DecisionEngine, Flag, Score
from klix.rules import Rule

# ============================================================================
# 1. CREATE THE ENGINE
# ============================================================================
# DecisionEngine(model_name=..., stop_words=...)
#   model_name : any FastEmbed-compatible embedding model
#                (default: multilingual MiniLM, ~120 MB, one-time download)
#   stop_words : None = built-in EN+DE list (default)
#                []    = stop-word filtering fully OFF
#                ["die","the",...] = custom list (e.g. a third language)
engine = DecisionEngine()
# engine = DecisionEngine(stop_words=[])                  # filtering off
# engine = DecisionEngine(stop_words=["der","die","the"])  # custom list

# ============================================================================
# 2. DEFINE THE HEADS - this IS the entire "training"
# ============================================================================
# Anchors = example sentences per class. More/better anchors = better quality.
# Languages may be mixed (DE/EN); the model is multilingual.

engine.add_head(
    Choice(
        name="route",
        options={
            # Invoicing / payment - German and English anchors mixed
            "billing": [
                "die rechnung wurde doppelt abgebucht",
                "gutschrift fehlt auf dem konto",
                "refund for my order is missing",
                "the invoice amount is wrong",
            ],
            # Technical malfunction
            "technical": [
                "der laptop startet nicht mehr",
                "vpn verbindung bricht staendig ab",
                "the server keeps crashing",
                "software update fails with an error",
            ],
            # Building / infrastructure
            "facility": [
                "die heizung im buero ist kaputt",
                "wasser tropft von der decke",
                "the elevator is stuck between floors",
                "parkplatz licht ist ausgefallen",
            ],
            # Security incident
            "security": [
                "verdächtiger login aus fremdem land",
                "ransomware hat den fileserver verschlüsselt",
                "suspicious login from another country",
                "phishing mail an alle mitarbeiter",
            ],
            # Explicit catch-all class: everything that belongs to no domain
            # queue. Advantage over plain value=None: such tickets get a real
            # bucket and can be closed automatically.
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
        # ---- Classifier choice --------------------------------------------
        # "nearest" (default): robust anchor comparison, good with few or
        #     mixed-language anchors.
        # "linear": learns a decision boundary (LogReg on the anchors),
        #     more accurate on single-language schemas with 3+ anchors/class.
        # "auto": picks for you - nearest for mixed languages, linear for
        #     single-language. Recommended if you don't want to decide manually.
        classifier="auto",
        #
        # ---- Optional: cross-lingual bridge -------------------------------
        # translate_fn mirrors every anchor into the missing language.
        # Signature: fn(text, target_lang) -> translated_text
        # A mini dictionary here for demo purposes; in production: a small
        # local translation model (e.g. OPUS-MT) or an LLM.
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
        # ---- Reject pole: texts that deserve NO class ---------------------
        # Those tickets are not routed but cleaned up:
        reject_anchors=[
            "danke für die tolle hilfe",
            "dickes lob an das support team",
            "freundliche grüße und schönes wochenende",
            "just saying hello and thanks",
        ],
        # ---- Score floor: too low a score -> also value=None --------------
        # (second safety layer for off-domain text without a reject pole)
        reject_threshold=0.35,
        #
        # ---- Rules: hard signals layered over the semantics (NEW in 0.4.0)
        # mode="force": if the pattern matches, the label wins IMMEDIATELY -
        #     even against semantics, the reject pole and reject_threshold.
        #     The most explicit user signal ("these words are always X").
        # mode="boost": adds `weight` to the label - can tip narrow decisions
        #     without forcing them.
        # Triggered rules appear later in details("route")["matched_rules"].
        rules=[
            # Ransomware/phishing/hacker keywords are ALWAYS a security issue,
            # no matter how the rest of the sentence sounds:
            Rule(
                label="security",
                pattern=r"(?i)\b(?:ransomware|phishing|hacker)\b",
                mode="force",
                name="security_keywords",
            ),
            # Enterprise pattern: unauthorized login attempts are ALWAYS
            # security, no matter which semantic bridge the embedding takes
            # (the "admin account" case otherwise risked landing in billing
            # via "Konto"). Critical patterns must be enforced
            # deterministically instead of trusting the semantics:
            Rule(
                label="security",
                pattern=r"(?i)(?:unbefugter|fremder|verdächtiger)\s+(?:login|zugriff)"
                        r"|\badmin[- ]?(?:account|konto)\b.*(?:eingeloggt|login)",
                mode="force",
                name="unauthorized_login",
            ),
            # Asset IDs (plc-xx) are always technical:
            Rule(
                label="technical",
                pattern=r"\bplc-\d+\b",
                mode="boost",
                weight=1.5,
                name="plc_asset_tag",
            ),
        ],
        #
        # ---- Other useful knobs -------------------------------------------
        # keyword_boost=0.5          : weight of exact word hits (asset IDs like "plc-34")
        # label_aggregation="topk"   : averages the best 2 anchors per class
        #                              (instead of max; useful from 4+ anchors/class)
        # classifier_C=10.0          : regularization of the linear probe
    )
)

engine.add_head(
    Score(
        name="urgency",
        # Score = a continuous axis between two poles.
        # Anchors below = calm, anchors above = urgent.
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
        min_val=0.0,       # lower end of the scale
        max_val=3.0,       # upper end of the scale
        aggregation="topk",  # robust against a single noisy anchor
        sharpness=8.0,     # steepness of the sigmoid curve (default 8.0)
        # ---- Coverage gate (NEW in 0.7.0): noisy scores are NOT passed on.
        # If the text matches neither pole (coverage < 0.3), the projection is
        # noise -> value=None + raw value in raw_value.
        min_coverage=0.3,
        # ---- Fallback for downstream systems (NEW in 0.7.2): --------------
        # Pipelines like Jira/Salesforce often cannot handle None.
        # Set a default here (e.g. a standard priority) that is passed through
        # instead of None. None = strict mode (default).
        fallback_value=None,
    )
)

engine.add_head(
    Flag(
        name="is_security",
        # Flag = yes/no decision with confidence signals.
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
        # Optional third pole: when "neutral" wins -> value=None
        # (instead of a guessed True/False on off-topic text).
        neutral_anchors=["routine anfrage", "allgemeine frage", "sonstiges thema"],
        threshold=0.5,  # cutoff for True (default 0.5)
        temp=0.12,      # softmax temperature; lower = sharper (default 0.12)
        # aggregation="topk"  # recommended from 3+ anchors per pole
    )
)

# ============================================================================
# 3. COMPILE (one-time, ~50-100 ms)
# ============================================================================
# Loads anchors, builds TF-IDF indices, augments and trains the probe if used.
# Without an explicit call this happens automatically on the first decide().
engine.compile()

# ============================================================================
# 4. MAKE DECISIONS
# ============================================================================
# Important: the text goes through the embedding model ONCE
# (measured on this workstation ~50-90 ms; hardware-dependent), all 3 heads
# then evaluate the same vectors (a few ms combined).

tickets = [
    "meine karte wurde doppelt belastet, bitte sofort prüfen!",   # -> billing
    "the server is down, production line stopped!",               # -> technical
    "die heizung im seminarraum geht nicht",                      # -> facility
    "jemand hat sich in den admin account eingeloggt, um 3 uhr",  # -> security
    "Ticket-4711: die kaffeemaschine hat nicht mehr genügend Pulver "
    "um unsere ganzen Mitarbeiter zu sättigen! Bitte schnell neues organisieren!",
    # -> not_relevant (coffee machine/procurement is none of the domain queues)
    "thanks a lot for the great support, have a nice weekend!",   # -> not_relevant
]

print("=" * 78)
print("klix-engine demo - 3 heads (Choice + Score + Flag), shared backbone")
print("=" * 78)

for ticket in tickets:
    res = engine.decide(ticket)

    # --- Accessing results -----------------------------------------------
    # 1) As an attribute (just the 'value' part):
    #       res.route / res.urgency / res.is_security
    # 2) As a full dict (all details):
    #       res.details("route") / res.details("urgency") / ...
    d_route = res.details("route")
    d_urg = res.details("urgency")
    d_flag = res.details("is_security")

    # ------------------------------------------------------------------
    # PRODUCTION PATTERN: two-stage decision
    # ------------------------------------------------------------------
    # A catch-all class ("not_relevant") is convenient but dangerous when it
    # wins with low confidence (then a real security case may lose against
    # it). Therefore:
    #   1. Security flag as a VETO: never auto-close a real incident
    #   2. Treat not_relevant automatically only at HIGH confidence
    #   3. everything else -> manual review
    CONF_GATE = 0.6  # confidence threshold for automation

    if d_route["value"] == "not_relevant" and res.is_security is True:
        action = "MANUAL REVIEW (security flag contradicts catch-all class!)"
    elif d_route["value"] != "not_relevant" and d_route["confidence"] >= CONF_GATE:
        action = f"-> route automatically into domain queue '{res.route}'"
    elif d_route["value"] == "not_relevant" and d_route["confidence"] >= CONF_GATE:
        action = "-> close automatically (not domain-relevant)"
    else:
        action = "-> MANUAL REVIEW (confidence too low for automation)"

    print(f"\nTicket : {ticket}")
    print(f"  {res!r}")
    print(f"  route       : {res.route!r:12s}  (score={d_route['score']:.2f}, "
          f"conf={d_route['confidence']:.2f}, reject={d_route.get('reject_score', 0):.2f})")
    # NEW in 0.7.0: coverage gate - noisy scores arrive as None + raw_value.
    # A CRM that only reads res.urgency can never process garbage again.
    if res.urgency is None:
        print(f"  urgency     : None (coverage gate: coverage={d_urg['coverage']:.2f} < 0.3, "
              f"raw value {d_urg.get('raw_value')} is noise and is NOT passed on)")
    else:
        print(f"  urgency     : {res.urgency:.2f}/3.0  (coverage={d_urg['coverage']:.2f})")
    print(f"  is_security : {res.is_security}  (p={d_flag['probability']:.2f}, "
          f"margin={d_flag['margin']:.2f}, coverage={d_flag['coverage']:.2f})")
    print(f"  >>> Action  : {action}")

# ============================================================================
# 5. INTERPRETING THE SIGNALS  (the heart of productive use)
# ============================================================================
print("\n" + "=" * 78)
print("Confidence signals, briefly explained")
print("=" * 78)
print("""
route  (Choice):
  score    - similarity to the best anchor (hybrid of dense + keyword)
  confidence - margin to the runner-up; low = borderline case
  reject_score - how strongly a reject anchor (small talk etc.) matches;
                 if it wins -> value=None
  All three together: only trust value if score is high AND
  confidence is not near 0 AND value is not None.

urgency (Score):
  value    - projection onto the low/high axis [min_val, max_val]
  coverage - similarity to the better-fitting pole.
             < ~0.3 = the text hit no pole -> the score is noise,
             do not use it! That is the honest "I don't know" signal.

is_security (Flag):
  probability - softmax confidence (NOT a calibrated probability!)
  margin      - |p_true - p_false|; large = clear decision
  coverage    - how strongly the text matches a pole at all
""")

# ============================================================================
# 6. CUSTOM HEADS  (subclass BaseHead)
# ============================================================================
print("=" * 78)
print("Custom head: RegExExtraction (extract the ticket number)")
print("=" * 78)

import re


class TicketIdHead(BaseHead):
    """Extracts IDs like 'INC-4711' or 'TICKET-42' from the text.

    Returns the ID WITH its original prefix -- an extraction head must not
    silently rewrite data (transformation rule).
    """

    def __init__(self, name: str = "ticket_id"):
        super().__init__(name)
        self.pattern = re.compile(r"(?i)\b(ticket|inc|case)[-# ]?(\d{2,6})\b")

    def get_reference_texts(self) -> list[str]:
        return []  # no reference texts needed

    def fit(self, backbone) -> None:
        pass  # nothing to precompute

    def evaluate(self, encoded) -> dict:
        m = self.pattern.search(encoded.text)
        if not m:
            return {"value": None}
        prefix, number = m.group(1).upper(), m.group(2)
        return {"value": f"{prefix}-{number}"}  # original prefix, no rewriting


engine.add_head(TicketIdHead())
engine.compile()  # new head -> recompile

demo = "Ticket-4711: die kaffeemaschine hat nicht mehr genügend Pulver um unsere ganzen Mitarbeiter zu sättigen! Bitte schnell neues organisieren!"
res = engine.decide(demo)
print(f"\nText  : {demo}")
print(f"route : {res.route} | ticket_id: {res.ticket_id} | urgency: {res.urgency}")

# ============================================================================
# 7. RULES IN ACTION - force and boost (NEW in 0.4.0)
# ============================================================================
print("\n" + "=" * 78)
print("Rules (force/boost) - hard signals layered over the semantics")
print("=" * 78)

rule_tickets = [
    # Force rule: "ransomware" forces security, whatever the semantics say
    ("phishing mail aussieht wie eine rechnung, bitte prüfen", "security forced"),
    # Boost rule: "plc-42" lifts technical (asset-ID signal)
    ("plc-42 sporadische fehler im schichtbetrieb", "technical via boost"),
    # No rule hit: plain semantics
    ("die heizung im besprechungsraum geht nicht", "routed normally"),
]
for text, expect in rule_tickets:
    res = engine.decide(text)
    d = res.details("route")
    matched = d.get("matched_rules", [])
    forced = d.get("forced_by")
    print(f"\nText : {text}")
    print(f"  route : {d['value']!r} (score={d['score']:.2f}, conf={d['confidence']:.2f})")
    if forced:
        print(f"  >>> FORCED by rule {forced!r}")
    elif matched:
        print(f"  >>> rule triggered: {matched}")
    else:
        print(f"  >>> no rule matched (purely semantic)")

# ============================================================================
# 8. EXPLAIN - WHY was it decided this way? (NEW in 0.4.0)
# ============================================================================
print("\n" + "=" * 78)
print("res.explain() - decision attribution (debugging)")
print("=" * 78)

explain_tickets = [
    "die rechnung wurde doppelt abgebucht, bitte sofort prüfen",  # clear billing case
    "jemand hat sich in den admin account eingeloggt, um 3 uhr",  # security
]
for text in explain_tickets:
    res = engine.decide(text)
    exp = res.explain("route")
    print(f"\nText : {text}")
    print(f"  decision: {exp['value']}")
    for b in exp.get("because", []):
        if b["kind"] == "semantic":
            print(f"    semantic : anchor {b['anchor']!r} -> {b['similarity']:.0%}")
        elif b["kind"] == "keyword":
            print(f"    keyword  : {b['token']!r} (contribution {b['weight']})")
        elif b["kind"] == "rule":
            print(f"    rule     : {b['rule']}")
    if exp.get("runner_up"):
        print(f"    runner-up: {exp['runner_up']['label']} ({exp['runner_up']['score']:.2f})")

# Flag and Score explain themselves too:
print()
exp_flag = engine.decide("ransomware verschlüsselt unsere daten!").explain("is_security")
print(f"Flag interpretation : {exp_flag['interpretation']}")
exp_score = engine.decide("produktion steht, sofort hilfe!").explain("urgency")
print(f"Score interpretation: {exp_score['interpretation']}")

# ============================================================================
# 9. CALIBRATE - learn thresholds instead of guessing (NEW in 0.4.0)
# ============================================================================
print("\n" + "=" * 78)
print("engine.calibrate() - learn the threshold from examples")
print("=" * 78)

flag_head = next(h for h in engine.heads if h.name == "is_security")
print(f"\nFlag threshold before: {flag_head.threshold}")
calib_samples = [
    ("ransomware hat den server verschlüsselt", True),
    ("fremder login im admin konto um 3 uhr", True),
    ("phishing mail an die buchhaltung", True),
    ("unbekannter datenabfluss gestern nacht", True),
    ("der drucker hat papierstau", False),
    ("der monitor flackert", False),
    ("das wlan ist langsam", False),
    ("maus kabel ist kaputt", False),
]
report = engine.calibrate("is_security", calib_samples)
print(f"Flag threshold after : {flag_head.threshold} "
      f"(metric={report['metric']}, value={report['value']:.2f}, n={report['n']})")
if report.get("warning"):
    print(f"  note: {report['warning']}")

# Score calibration: learn sharpness + affine remap from target values
print("\nScore calibration (target values instead of the default scale):")
score_samples = [
    ("produktion steht komplett still", 3.0),
    ("notfall, alles fällt aus", 3.0),
    ("kritischer ausfall läuft gerade", 2.7),
    ("routinefrage, keine eile", 0.2),
    ("kann bis nächste woche warten", 0.1),
    ("normale anfrage ohne priorität", 0.3),
]
report = engine.calibrate("urgency", score_samples)
print(f"  sharpness={report['sharpness']}, remap: value ≈ {report['a']:.2f} + {report['b']:.2f}·raw")
check = engine.decide("produktion steht komplett still, alles down!")
print(f"  check 'produktion steht komplett still, sofort': urgency={check.urgency} (target ~3.0)")
check2 = engine.decide("routinefrage, kann warten")
print(f"  check 'routinefrage, kann warten': urgency={check2.urgency} (target ~0.2)")

# ============================================================================
# 10. BATCH - bulk processing (NEW in 0.5.0)
# ============================================================================
print("\n" + "=" * 78)
print("engine.decide_batch() - one embedding pass for all texts")
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
print(f"\n{len(batch_texts)} texts in one embedding pass: {batch_ms:.1f} ms total "
      f"({batch_ms/len(batch_texts):.2f} ms/item)")
print("Note: batching amortizes mainly the fixed call overhead of the model;")
print("the ONNX forward itself scales ~linearly with the token count. Measured")
print("(evals/bench_batch.py):")
print("  n=5: ~1.5x | n=25: ~3.1x | n=500: ~2.5-3.2x faster than serial.")
print("  Rule of thumb: decide() for single requests, decide_batch() from ~25 texts.")
for text, res in zip(batch_texts, batch):
    print(f"  {res.route!r:15s} <- {text}")

# ============================================================================
# 11. VALIDATE_ANCHORS - class-overlap report (NEW in 0.6.0)
# ============================================================================
print("\n" + "=" * 78)
print("engine.validate_anchors_report() - check anchor quality")
print("=" * 78)
print("\nRead-only diagnosis: overlapping classes, confuser words,")
print("misplaced and duplicate anchors. Never mutates anchors.\n")

# Second engine with deliberately overlapping classes to show the report:
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
print("\n(legend: !! = high overlap, ? = medium; 'sharpen' = exclusive terms)")
print(" that would sharpen the respective class)")

# And show that the MAIN engine is clean:
main_report = engine.validate_anchors_report()
print("\nMain engine:", main_report.splitlines()[0] if main_report else "no findings")

# ============================================================================
# 12. LATENCY PROFILE
# ============================================================================
print("\n" + "=" * 78)
print("Latency profile (50 runs, median)")
print("=" * 78)
engine.decide("warmup")
times = []
for _ in range(50):
    t0 = time.perf_counter()
    engine.decide("plc-34 meldet fehler, förderband steht sofort")
    times.append((time.perf_counter() - t0) * 1000)
print(f"  3 heads + TicketIdHead: {sorted(times)[25]:.1f} ms median "
      f"(of which ~50-90 ms embedding pass, head math a few ms)")

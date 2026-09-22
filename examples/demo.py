"""Demo: Klix-Engine mit drei Köpfen (Choice, Score, Flag) auf IT/OT-Tickets."""

import time

from klix import DecisionEngine, Flag, Score, Choice


def main() -> None:
    engine = DecisionEngine()

    engine.add_head(
        Choice(
            name="target",
            options={
                "it_ops": ["VPN abgerissen", "Server down", "Rechner bootet nicht", "web-02 timeout"],
                "ot_plant": ["Roboterzelle steht", "SPS Fehler", "plc-34 fehler", "Taktzeit deviation"],
                "finance": ["KST 4210 über Budget", "Rechnung freigeben", "Skonto abziehen"],
                "facility": ["Schmiermittel-Pfütze Rutschgefahr", "Öllache Halle 2", "Heizung defekt"],
            },
        )
    )

    engine.add_head(
        Score(
            name="urgency",
            low_anchors=["Routine-Wartung", "Informelle Frage", "Hat Zeit nächste Woche"],
            high_anchors=["Notfall sofort", "Produktionsstillstand", "Akute Gefahr", "Kritischer Ausfall"],
            min_val=0.0,
            max_val=3.0,
        )
    )

    engine.add_head(
        Flag(
            name="is_security",
            true_anchors=["Hackerangriff", "Ransomware Befall", "Root login kompromittiert", "Datenabfluss"],
            false_anchors=["Hardware kaputt", "Standard IT Problem", "Netzwerkstörung", "Alltägliche Anfrage"],
            neutral_anchors=["Routineanfrage", "Allgemeine Frage", "Sonstiges Thema"],
            threshold=0.5,
        )
    )

    engine.compile()

    tickets = [
        "plc-34 meldet fehler, förderband steht sofort!",
        "Kaffee ist alle, wer füllt die Maschine nach?",
        "Halle 2 neben der Presse steht Öl auf dem Boden, jemand ist fast ausgerutscht",
        "Verdächtiger Login auf dem Domain-Controller, Root-Zugriff um 3 Uhr nachts",
        "VPN bricht bei Homeoffice ständig ab, Kundentermin in 10 Minuten",
        "Rechnung 2024-118 bitte freigeben, Skonto läuft morgen ab",
        "Roboterzelle 3 bleibt im Zyklus stehen, Taktzeit verdoppelt",
        "Routine-Wartung der Lüftung ist nächste Woche geplant",
    ]

    print("=" * 72)
    print("Klix-Engine Demo - 3 Köpfe, geteilter Backbone")
    print("=" * 72)

    for ticket in tickets:
        result = engine.decide(ticket)
        security = result.details("is_security")
        print(f"\nTicket : {ticket}")
        print(f"Result : {result}")
        print(
            f"  target={result.target} (Konfidenz {result.details('target')['confidence']:.0%}) | "
            f"urgency={result.urgency}/3.0 | "
            f"is_security={result.is_security} (p={security['probability']:.1%})"
        )

    # Latenz-Test: 10 Durchläufe, zeigt Shared-Backbone-Kosten vs. Kopf-Kosten.
    print("\n" + "=" * 72)
    print("Latenzprofil (10 Durchläufe je Ticket)")
    print("=" * 72)
    total = 0.0
    for ticket in tickets[:3]:
        times = []
        for _ in range(10):
            result = engine.decide(ticket)
            times.append(result.latency_ms)
        avg = sum(times) / len(times)
        total += avg
        print(f"  {avg:6.1f} ms Ø  |  {ticket[:50]}")
    print(f"\nGeteilter Backbone: 8 Tickets × 3 Köpfe = 24 Kopf-Entscheidungen bei "
          f"nur {total / 3:.0f} ms Ø Embedding-Kosten pro Durchlauf.")


if __name__ == "__main__":
    main()
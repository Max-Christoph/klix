"""Demo: Klix engine with three heads (Choice, Score, Flag) on IT/OT tickets."""

import time

from klix import DecisionEngine, Flag, Score, Choice


def main() -> None:
    engine = DecisionEngine()

    engine.add_head(
        Choice(
            name="target",
            options={
                "it_ops": ["VPN down", "server unreachable", "laptop won't boot", "web-02 timeout"],
                "ot_plant": ["robot cell stopped", "PLC fault", "plc-34 error", "cycle time deviation"],
                "finance": ["cost center 4210 over budget", "approve invoice", "apply early-payment discount"],
                "facility": ["lubricant puddle slip hazard", "oil spill hall 2", "heating broken"],
            },
        )
    )

    engine.add_head(
        Score(
            name="urgency",
            low_anchors=["routine maintenance", "casual question", "can wait until next week"],
            high_anchors=["emergency right now", "production line down", "acute danger", "critical outage"],
            min_val=0.0,
            max_val=3.0,
        )
    )

    engine.add_head(
        Flag(
            name="is_security",
            true_anchors=["hacker attack", "ransomware infection", "compromised root login", "data exfiltration"],
            false_anchors=["hardware broken", "standard IT problem", "network outage", "everyday request"],
            neutral_anchors=["routine request", "general question", "other topic"],
            threshold=0.5,
        )
    )

    engine.compile()

    tickets = [
        "plc-34 reports a fault, conveyor belt stopped immediately!",
        "coffee machine is empty, who will refill it?",
        "hall 2 next to the press: oil on the floor, someone almost slipped",
        "suspicious login on the domain controller, root access at 3 am",
        "VPN keeps dropping in home office, customer call in 10 minutes",
        "please approve invoice 2024-118, early-payment discount expires tomorrow",
        "robot cell 3 stalls mid-cycle, cycle time doubled",
        "routine ventilation maintenance scheduled for next week",
    ]

    print("=" * 72)
    print("Klix engine demo - 3 heads, shared backbone")
    print("=" * 72)

    for ticket in tickets:
        result = engine.decide(ticket)
        security = result.details("is_security")
        print(f"\nTicket : {ticket}")
        print(f"Result : {result}")
        print(
            f"  target={result.target} (confidence {result.details('target')['confidence']:.0%}) | "
            f"urgency={result.urgency}/3.0 | "
            f"is_security={result.is_security} (p={security['probability']:.1%})"
        )

    # Latency test: 10 runs per ticket, shows shared-backbone cost vs. head cost.
    print("\n" + "=" * 72)
    print("Latency profile (10 runs per ticket)")
    print("=" * 72)
    total = 0.0
    for ticket in tickets[:3]:
        times = []
        for _ in range(10):
            result = engine.decide(ticket)
            times.append(result.latency_ms)
        avg = sum(times) / len(times)
        total += avg
        print(f"  {avg:6.1f} ms avg  |  {ticket[:50]}")
    print(f"\nShared backbone: 8 tickets x 3 heads = 24 head decisions at only "
          f"{total / 3:.0f} ms avg embedding cost per pass.")


if __name__ == "__main__":
    main()
"""Evaluation harness: measures routing accuracy, score error, and flag behavior
on labeled cases. Run: uv run python evals/eval_baseline.py
"""

import sys
import time

sys.path.insert(0, "src")

from klix import DecisionEngine, Choice, Score, Flag  # noqa: E402

# ---------------------------------------------------------------------------
# Labeled test cases. `label` = expected head outcome.
# Choice options deliberately use DIFFERENT example sentences than the queries
# (no leakage: the engine must generalize, not memorize).
# ---------------------------------------------------------------------------

CHOICE_OPTIONS = {
    "it_ops": [
        "cannot connect to the VPN",
        "the server keeps crashing",
        "laptop will not boot",
        "email client shows an error",
    ],
    "ot_plant": [
        "the robot cell stopped mid cycle",
        "PLC reports a sensor fault",
        "conveyor belt does not move",
        "machine throws error code",
    ],
    "finance": [
        "please approve this invoice",
        "cost center budget exceeded",
        "travel expense reimbursement",
        "purchase order needs release",
    ],
    "facility": [
        "water is leaking in the corridor",
        "heating does not warm up",
        "someone spilled oil on the floor",
        "the light switches are broken",
    ],
}

SCORE_LOW = ["routine maintenance", "casual question", "can wait until next week", "no rush at all"]
SCORE_HIGH = ["emergency right now", "production line down", "acute danger", "critical outage"]

FLAG_TRUE = ["hacker attack", "ransomware infection", "compromised root login", "data exfiltration"]
FLAG_FALSE = ["hardware broken", "ordinary IT problem", "network outage", "everyday request"]
FLAG_NEUTRAL = ["routine request", "general question", "other topic"]

# (query, expected_label) — paraphrases, typos, mixed language, OOD
CHOICE_CASES = [
    ("the vpn tunnel collapses every few minutes", "it_ops"),
    ("my notebook refuses to start since this morning", "it_ops"),
    ("outlook throws send and receive error", "it_ops"),
    ("the conveyer belt is jammed again", "ot_plant"),
    ("robot arm halts with servo error", "ot_plant"),
    ("sps meldet stoerung an station 4", "ot_plant"),  # German query, English anchors
    ("plc-12 reports fault code 402", "ot_plant"),
    ("expense report approval pending since friday", "finance"),
    ("department 4210 blew its budget again", "finance"),
    ("the invoice for the new chairs needs sign-off", "finance"),
    ("there is a water puddle next to the elevator", "facility"),
    ("radiators stay cold in meeting room 2", "facility"),
    ("oil streak between the machines", "facility"),
    ("flip the light swich in hall 3", "facility"),  # typo
    ("who stole the last yogurt from the fridge", None),  # out of domain
    ("happy birthday to the whole team", None),  # out of domain
]

SCORE_CASES = [
    ("production line is down, we lose thousands per minute", 2.5, 3.0),
    ("emergency! machine caught fire, evacuate", 2.5, 3.0),
    ("conveyor blocked, stop the line now", 2.0, 3.0),
    ("someone should fix the printer this week", 0.0, 1.5),
    ("whats the wifi password", 0.0, 1.0),
    ("no rush, whenever you find time next month", 0.0, 1.0),
    ("customer demo fails in ten minutes, need the demo env now", 2.0, 3.0),
    ("planning the summer party activities", 0.0, 1.0),
]

FLAG_CASES = [
    ("someone ran ransomware on our file server", True),
    ("unknown admin logged in at 3am and copied customer data", True),
    ("brute force attack on our ssh gateway", True),
    ("printer cartridge is empty", False),
    ("the monitor flickers sometimes", False),
    ("wifi is slow in meeting room 1", False),
    ("please order more coffee beans", False),
]


def build_engine() -> DecisionEngine:
    eng = DecisionEngine()
    eng.add_head(Choice(name="target", options=CHOICE_OPTIONS))
    eng.add_head(Score(name="urgency", low_anchors=SCORE_LOW, high_anchors=SCORE_HIGH,
                       min_val=0.0, max_val=3.0))
    eng.add_head(Flag(name="is_security",
                      true_anchors=FLAG_TRUE, false_anchors=FLAG_FALSE,
                      neutral_anchors=FLAG_NEUTRAL, threshold=0.5))
    eng.compile()
    return eng


def run_eval(eng: DecisionEngine, verbose: bool = False) -> dict:
    choice_ok = 0
    ood_rejected = 0
    ood_total = 0
    choice_errors = []
    score_errors = []
    score_worst = []
    flag_results = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "none": 0}
    confusions = []

    for query, expected in CHOICE_CASES:
        res = eng.decide(query)
        got = res.target
        if expected is None:
            ood_total += 1
            details = res.details("target")
            # OOD "correct" = very low confidence (engine should not be sure)
            if details["score"] < 0.35 and details["confidence"] < 0.5:
                ood_rejected += 1
            elif verbose:
                choice_errors.append((query, got, f"score={details['score']:.2f} conf={details['confidence']:.2f}"))
        elif got == expected:
            choice_ok += 1
        else:
            choice_errors.append((query, expected, got))
            confusions.append(f"{expected}->{got}")

    for query, lo, hi in SCORE_CASES:
        res = eng.decide(query)
        got = res.urgency
        err = abs(got - (lo + hi) / 2)
        ok = lo <= got <= hi
        if not ok:
            score_errors.append((query, f"[{lo}-{hi}]", got))
            score_worst.append((err, query, got))

    for query, expected in FLAG_CASES:
        prob = eng.decide(query).details("is_security")["probability"]
        value = prob >= 0.5
        if value and prob >= 0.5:
            if expected:
                flag_results["tp"] += 1
            else:
                flag_results["fp"] += 1
        elif expected:
            flag_results["fn"] += 1
        else:
            flag_results["tn"] += 1

    n_choice = len([c for c in CHOICE_CASES if c[1] is not None])
    return {
        "choice_acc": choice_ok / n_choice,
        "choice_errors": choice_errors,
        "ood_rejected": f"{ood_rejected}/{ood_total}",
        "score_errors": score_errors,
        "score_worst": sorted(score_worst, reverse=True)[:3],
        "flag": flag_results,
        "confusions": confusions,
    }


def main():
    eng = build_engine()
    # Warmup
    eng.decide("warmup")

    r = run_eval(eng, verbose=True)

    print("=" * 70)
    print("BASELINE EVALUATION")
    print("=" * 70)
    print(f"Choice accuracy (in-domain): {r['choice_accuracy']:.0%}" if False else
          f"Choice accuracy (in-domain): {r['choice_acc']:.0%}")
    print(f"OOD rejection:               {r['ood_rejected']}")
    n_flag = sum(r["flag"].values())
    tp, tn, fp, fn = r["flag"]["tp"], r["flag"]["tn"], r["flag"]["fp"], r["flag"]["fn"]
    print(f"Flag: TP={tp} TN={tn} FP={fp} FN={fn}")
    if r["choice_errors"]:
        print("\nChoice errors:")
        for q, exp, got in r["choice_errors"]:
            print(f"  [{exp}] -> [{got}]: {q}")
    if r["score_errors"]:
        print("\nScore violations (outside expected band):")
        for q, band, got in r["score_errors"]:
            print(f"  {band}: got {got} | {q}")
    if r["score_worst"]:
        print("\nWorst score deviations:")
        for err, q, got in r["score_worst"]:
            print(f"  err={err:.2f} got={got:.2f} | {q}")


if __name__ == "__main__":
    main()
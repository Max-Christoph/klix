"""Cross-domain evaluation: image captions, task management, e-commerce support.

Tests:
1. Each domain standalone (accuracy on labeled cases).
2. All domains in ONE engine (decoupling claim: identical results).
3. OOD rejection and coverage behavior per domain.
4. Multilingual queries (German captions against English anchors).

Run: uv run python -c "import sys; sys.path.insert(0,'.'); sys.path.insert(0,'src'); import evals.eval_domains"
"""

import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from klix import DecisionEngine, Choice, Score, Flag  # noqa: E402

# ---------------------------------------------------------------------------
# Domain A: image description triage (photo queue / moderation)
# ---------------------------------------------------------------------------
IMG_OPTIONS = {
    "nature": ["sunset over the mountains", "forest path with morning fog", "lake reflecting the sky"],
    "people": ["group of colleagues celebrating", "child playing in the garden", "portrait of a smiling woman"],
    "document": ["crumpled supermarket receipt", "screenshot of a spreadsheet", "signed contract on a desk"],
    "food": ["bowl of ramen on a wooden table", "birthday cake with candles", "fresh vegetables at the market"],
    "vehicle": ["red sports car on a highway", "tractor working in the field", "bicycle leaning on a wall"],
}
IMG_QUALITY_LOW = ["blurry out of focus snapshot", "very dark underexposed photo", "accidental pocket shot"]
IMG_QUALITY_HIGH = ["sharp professional studio photograph", "well lit crisp composition", "highly detailed panorama"]
IMG_PERSON_TRUE = ["close-up of a visible face", "crowd of pedestrians walking", "family posing for the camera"]
IMG_PERSON_FALSE = ["empty landscape with no people", "close-up of a coffee cup", "screenshot of a web page"]
IMG_PERSON_NEUTRAL = ["heavily blurred motion shot", "object photographed from above", "abstract gradient artwork"]

IMG_CASES = [
    ("a man riding a bike through the park", "vehicle"),
    ("misty mountain ridge at sunrise", "nature"),
    ("invoice photographed on a wooden table", "document"),
    ("plate of sushi rolls close up", "food"),
    ("two kids building a sandcastle", "people"),
    ("delivery van parked at the loading dock", "vehicle"),
    ("waterfall in a green valley", "nature"),
    ("keyboard and monitor on an office desk, windows visible", "document"),
    ("ein nebeliger waldweg am fruhen morgen", "nature"),  # German
    ("team lunch at the new italian restaurant", "food"),
    ("slightly blurry photo of a distant bird", "nature"),
    ("annual report lying open on the table", "document"),
]
IMG_OOD = ["quarterly revenue increased by four percent", "please forward this email to legal"]
IMG_QUALITY_BANDS = [
    ("razor sharp golden hour landscape, perfect lighting", 2.0, 3.0),
    ("pixelated low light mess", 0.0, 1.2),
    ("decent sharp snapshot of a dog", 1.0, 2.6),
    ("blurred shaky night video still", 0.0, 1.2),
]
IMG_PERSON_FLAG = [
    ("smiling child holding an ice cream cone, face clearly visible", True),
    ("seaside cliffs without a single person", False),
    ("close-up portrait, eyes in focus", True),
    ("empty parking lot at dawn", False),
]

# ---------------------------------------------------------------------------
# Domain B: task inbox (task routing)
# ---------------------------------------------------------------------------
TASK_OPTIONS = {
    "email_action": ["reply to the customer proposal", "forward the minutes to the team", "answer the support thread"],
    "scheduling": ["book a meeting room for thursday", "find a slot for the sync", "move the standup to friday"],
    "errand": ["pick up the package at the station", "buy printer paper on the way home", "water the office plants"],
    "coding": ["fix the failing unit test", "refactor the auth module", "review the pull request"],
    "research": ["look up competitor pricing", "read the new whitepaper", "summarize the survey results"],
}
TASK_EFFORT_LOW = ["two minute quick reply", "five minute errand", "one liner fix"]
TASK_EFFORT_HIGH = ["multi week project", "requires days of focused work", "whole team for a month"]
TASK_DELEGATE_TRUE = ["anyone from the office can do this", "delegate to the assistant", "simple task for an intern"]
TASK_DELEGATE_FALSE = ["requires my personal signature", "needs my specific expertise", "only the board may approve"]
TASK_DELEGATE_NEUTRAL = ["optional nice to have", "no deadline attached", "whenever convenient"]

TASK_CASES = [
    ("please respond to the rfp email from yesterday", "email_action"),
    ("schedule the quarterly business review", "scheduling"),
    ("collect the parcel from the post office", "errand"),
    ("the build is red, fix the broken test", "coding"),
    ("investigate what drives our churn rate", "research"),
    ("sign the rental agreement for the new office", "email_action"),
    ("reschedule the design review to friday", "scheduling"),
    ("grab coffee beans for the kitchen", "errand"),
    ("update the deprecated dependency in the backend", "coding"),
    ("find studies about remote work productivity", "research"),
    ("termin fur die jahresplanung finden", "scheduling"),  # German
    ("write back to the recruiter about the interview", "email_action"),
]
TASK_EFFORT_BANDS = [
    ("rewrite the entire billing system", 2.0, 3.0),
    ("answer yes or no to the invite", 0.0, 1.0),
    ("migrate the database to the new cluster", 2.0, 3.0),
    ("forward the memo to the team", 0.0, 1.0),
]
TASK_DELEGATE_FLAG = [
    ("order catering, anyone can place the call", True),
    ("sign the contract personally, nobody else may", False),
    ("have the intern sort the archive", True),
    ("my signature is legally required here", False),
]

# ---------------------------------------------------------------------------
# Domain C: e-commerce customer support
# ---------------------------------------------------------------------------
SHOP_OPTIONS = {
    "refund": ["i want my money back", "return the defective item for a refund", "cancel and reimburse the order"],
    "shipping": ["where is my package", "delivery is late", "track my parcel"],
    "product": ["does this shirt come in blue", "what are the dimensions of the shelf", "is the laptop compatible with linux"],
    "complaint": ["your service is a disaster", "fifth broken item in a row", "never again buying here"],
}
SHOP_SENTIMENT_LOW = ["polite friendly inquiry", "calm neutral question", "curious customer"]
SHOP_SENTIMENT_HIGH = ["furious customer threatening lawyer", "scathing public review", "enraged rant"]
SHOP_CHURN_TRUE = ["canceling my subscription immediately", "switching to your competitor", "closing my account for good"]
SHOP_CHURN_FALSE = ["happy long term customer", "looking forward to the delivery", "just browsing the sale"]
SHOP_CHURN_NEUTRAL = ["first time visitor", "one off purchase", "gift for a friend"]

SHOP_CASES = [
    ("please reimburse order 4482, the shoes do not fit", "refund"),
    ("my parcel has not arrived after two weeks", "shipping"),
    ("does the sofa table also come in walnut", "product"),
    ("the delivery driver never even rang the doorbell", "complaint"),
    ("i returned the blender, when do i get the refund", "refund"),
    ("tracking shows the box in the wrong city", "shipping"),
    ("how long is the warranty on the mixer", "product"),
    ("the packaging arrived completely destroyed", "complaint"),
    ("wohne bleibt mein paket liegen", "shipping"),  # German (typo'd)
    ("can i exchange the sweater for a larger size", "refund"),
    ("ist the toaster dual voltage for the us", "product"),
    ("waiting three days for a status update now", "complaint"),
]
SHOP_SENTIMENT_BANDS = [
    ("i am absolutely done with you people, see you in court", 2.0, 3.0),
    ("hello, quick question about sizes please", 0.0, 1.0),
    ("this is the third broken lamp, unacceptable!!", 2.0, 3.0),
    ("thanks for the quick reply last time", 0.0, 1.0),
]
SHOP_CHURN_FLAG = [
    ("i will terminate my premium account today", True),
    ("great shop, will order again next month", False),
    ("moving my whole team to the competitor platform", True),
    ("just a birthday present for my sister", False),
]


def build_standalone(domain: str) -> tuple[DecisionEngine, list]:
    eng = DecisionEngine()
    if domain == "image":
        eng.add_head(Choice(name="category", options=IMG_OPTIONS, keyword_boost=0.3))
        eng.add_head(Score(name="quality", low_anchors=IMG_QUALITY_LOW, high_anchors=IMG_QUALITY_HIGH,
                           min_val=0.0, max_val=3.0, aggregation="topk"))
        eng.add_head(Flag(name="shows_person", true_anchors=IMG_PERSON_TRUE,
                          false_anchors=IMG_PERSON_FALSE, neutral_anchors=IMG_PERSON_NEUTRAL))
        cases = (IMG_CASES, IMG_QUALITY_BANDS, IMG_PERSON_FLAG, IMG_OOD)
    elif domain == "task":
        eng.add_head(Choice(name="kind", options=TASK_OPTIONS))
        eng.add_head(Score(name="effort", low_anchors=TASK_EFFORT_LOW, high_anchors=TASK_EFFORT_HIGH,
                           min_val=0.0, max_val=3.0, aggregation="topk"))
        eng.add_head(Flag(name="delegatable", true_anchors=TASK_DELEGATE_TRUE,
                          false_anchors=TASK_DELEGATE_FALSE, neutral_anchors=TASK_DELEGATE_NEUTRAL))
        cases = (TASK_CASES, TASK_EFFORT_BANDS, TASK_DELEGATE_FLAG)
    else:  # shop
        eng.add_head(Choice(name="intent", options=SHOP_OPTIONS))
        eng.add_head(Score(name="sentiment", low_anchors=SHOP_SENTIMENT_LOW, high_anchors=SHOP_SENTIMENT_HIGH,
                           min_val=0.0, max_val=3.0, aggregation="topk"))
        eng.add_head(Flag(name="churn_risk", true_anchors=SHOP_CHURN_TRUE,
                          false_anchors=SHOP_CHURN_FALSE, neutral_anchors=SHOP_CHURN_NEUTRAL))
        cases = (SHOP_CASES, SHOP_SENTIMENT_BANDS, SHOP_CHURN_FLAG)
    eng.compile()
    return eng, cases


IMG_OOD = ["quarterly revenue increased by four percent", "please water my plants while on holiday"]


def eval_domain(eng: DecisionEngine, cases: tuple, choice_name: str, score_name: str, flag_name: str) -> dict:
    choice_cases, score_bands, flag_cases = cases[:3]
    ok = 0
    errors = []
    for query, expected in choice_cases:
        got = getattr(eng.decide(query), choice_name)
        if got == expected:
            ok += 1
        else:
            errors.append((query, expected, got))
    n = len(choice_cases)

    s_ok = 0
    s_flagged = 0
    s_errors = []
    for query, lo, hi in score_bands:
        d = eng.decide(query).details(score_name)
        got, cov = d["value"], d["coverage"]
        if cov < 0.30:
            s_flagged += 1
        elif lo <= got <= hi:
            s_ok += 1
        else:
            s_errors.append((query, f"[{lo}-{hi}]", got))
    nb = len(score_bands)

    f_ok = 0
    f_errors = []
    for query, expected in flag_cases:
        prob = eng.decide(query).details(flag_name)["probability"]
        if (prob >= 0.5) == expected:
            f_ok += 1
        else:
            f_errors.append((query, expected, round(prob, 2)))
    nf = len(flag_cases)

    return {
        "choice": f"{ok}/{n}", "choice_errors": errors,
        "score": f"{s_ok}/{nb} in-band, {s_flagged} low-coverage", "score_errors": s_errors,
        "flag": f"{f_ok}/{nf}", "flag_errors": f_errors,
    }


def main():
    print("=" * 74)
    print("CROSS-DOMAIN EVALUATION (image captions | task inbox | e-commerce support)")
    print("=" * 74)

    report = {}
    for domain, cn, sn, fn in [("image", "category", "quality", "shows_person"),
                               ("task", "kind", "effort", "delegatable"),
                               ("shop", "intent", "sentiment", "churn_risk")]:
        eng, cases = build_standalone(domain)
        eng.decide("warmup")
        r = eval_domain(eng, cases, cn, sn, fn)
        report[domain] = r
        print(f"\n--- {domain.upper()} ---")
        print(f"Choice : {r['choice']}")
        for q, exp, got in r["choice_errors"]:
            print(f"   ERROR [{exp}] -> [{got}] | {q}")
        print(f"Score  : {r['score']}")
        for q, band, got in r["score_errors"]:
            print(f"   VIOLATION {band}: got {got} | {q}")
        print(f"Flag   : {r['flag']}")
        for q, exp, prob in r["flag_errors"]:
            print(f"   ERROR expected {exp}, p={prob} | {q}")

    # Decoupling test: all three domains in ONE engine, identical results?
    print("\n" + "=" * 74)
    print("DECOUPLING TEST: all domains in one engine (results must match standalone)")
    print("=" * 74)
    mega = DecisionEngine()
    mega.add_head(Choice(name="img_category", options=IMG_OPTIONS, keyword_boost=0.3))
    mega.add_head(Score(name="img_quality", low_anchors=IMG_QUALITY_LOW, high_anchors=IMG_QUALITY_HIGH,
                        min_val=0.0, max_val=3.0, aggregation="topk"))
    mega.add_head(Choice(name="task_kind", options=TASK_OPTIONS))
    mega.add_head(Score(name="task_effort", low_anchors=TASK_EFFORT_LOW, high_anchors=TASK_EFFORT_HIGH,
                        min_val=0.0, max_val=3.0, aggregation="topk"))
    mega.add_head(Choice(name="shop_intent", options=SHOP_OPTIONS))
    mega.add_head(Score(name="shop_sentiment", low_anchors=SHOP_SENTIMENT_LOW, high_anchors=SHOP_SENTIMENT_HIGH,
                        min_val=0.0, max_val=3.0, aggregation="topk"))
    mega.compile()

    mismatches = 0
    checks = 0
    for domain, cn, sn in [("image", "category", "quality"), ("task", "kind", "effort"), ("shop", "intent", "sentiment")]:
        eng, cases = build_standalone(domain)
        choice_cases, score_bands = cases[0], cases[1]
        for query, expected in choice_cases:
            standalone = getattr(eng.decide(query), cn)
            combined = getattr(mega.decide(query),
                               {"image": "img_category", "task": "task_kind", "shop": "shop_intent"}[domain])
            checks += 1
            if standalone != combined:
                mismatches += 1
                print(f"  MISMATCH {domain}: standalone={standalone} combined={combined} | {query}")
        for query, lo, hi in score_bands:
            standalone = eng.decide(query).details(sn)["value"]
            combined = mega.decide(query).details(
                {"image": "img_quality", "task": "task_effort", "shop": "shop_sentiment"}[domain]
            )["value"]
            checks += 1
            if abs(standalone - combined) > 1e-9:
                mismatches += 1
                print(f"  MISMATCH {domain} score: standalone={standalone} combined={combined} | {query}")
    print(f"Decoupling: {mismatches}/{checks} mismatches (0 = perfect isolation)")

    # OOD rejection per domain (choice with reject anchors not configured here,
    # so expect guessing; coverage is the signal for scores)
    print("\n--- OOD score coverage per domain (lower = more honest) ---")
    eng, cases = build_standalone("image")
    for q in IMG_OOD:
        d = eng.decide(q).details("quality")
        print(f"  OOD cov={d['coverage']:.3f} -> {d['value']:.2f} | {q}")

    # Latency
    start = time.perf_counter()
    for _ in range(20):
        mega.decide("refill the coffee machine in the break room")
    ms = (time.perf_counter() - start) * 1000 / 20
    print(f"\nLatency (6 heads, one engine): {ms:.1f} ms per decide (encoding dominates)")


if __name__ == "__main__":
    main()
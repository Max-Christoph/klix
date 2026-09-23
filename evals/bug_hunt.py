"""Systematic edge-case hunt across all 0.4-0.6 features. Not a test file —
a bug-hunting script that exercises unusual inputs and prints anomalies."""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from klix import BaseHead, Choice, DecisionEngine, Flag, Score
from klix.rules import Rule

issues = []

def check(name, fn):
    try:
        fn()
        print(f"  OK    {name}")
    except Exception as e:
        issues.append((name, repr(e)))
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")

# ---------------------------------------------------------------- engine core
def t_empty_string():
    eng = DecisionEngine()
    eng.add_head(Choice(name="c", options={"a": ["alpha one"], "b": ["beta one"]}))
    eng.add_head(Score(name="s", low_anchors=["calm"], high_anchors=["crisis"]))
    eng.add_head(Flag(name="f", true_anchors=["attack"], false_anchors=["printer"]))
    eng.compile()
    r = eng.decide("")
    assert r.c is not None or r.c is None  # must not raise

def t_whitespace_only():
    eng = DecisionEngine()
    eng.add_head(Choice(name="c", options={"a": ["alpha one"], "b": ["beta one"]}))
    eng.compile()
    r = eng.decide("   ")
    assert True

def t_unicode_emoji():
    eng = DecisionEngine()
    eng.add_head(Choice(name="c", options={"a": ["alpha"], "b": ["beta"]}))
    eng.compile()
    r = eng.decide("🚨🔥 αlpha 💥")

def t_determinism():
    eng = DecisionEngine()
    eng.add_head(Choice(name="c", options={"a": ["alpha one two"], "b": ["beta one two"]}))
    eng.compile()
    vals = {eng.decide("alpha probe").c for _ in range(5)}
    assert len(vals) == 1, f"nondeterministic: {vals}"

def t_batch_mixed_lengths():
    eng = DecisionEngine()
    eng.add_head(Choice(name="c", options={"a": ["alpha"], "b": ["beta"]}))
    eng.compile()
    out = eng.decide_batch(["a", "", "very long " * 100, "🙂"])
    assert len(out) == 4

def t_batch_very_large():
    eng = DecisionEngine()
    eng.add_head(Choice(name="c", options={"a": ["alpha"], "b": ["beta"]}))
    eng.compile()
    out = eng.decide_batch([f"text {i}" for i in range(500)])
    assert len(out) == 500

# ---------------------------------------------------------------- rules
def t_rule_empty_text():
    eng = DecisionEngine()
    eng.add_head(Choice(name="c", options={"a": ["alpha"], "b": ["beta"]},
                        rules=[__import__("klix").rules.Rule(label="a", any_of=["x"], mode="force")]))
    eng.compile()
    assert eng.decide("").c != "a"

def t_rule_case_insensitive():
    eng = DecisionEngine()
    eng.add_head(Choice(name="c", options={"a": ["alpha"], "b": ["beta"]},
                        rules=[__import__("klix").rules.Rule(label="a", any_of=["NOTFALL"], mode="force")]))
    eng.compile()
    assert eng.decide("großer NOTFALL jetzt").c == "a"

def t_rule_regex_special_chars():
    eng = DecisionEngine()
    eng.add_head(Choice(name="c", options={"a": ["alpha"], "b": ["beta"]},
                        rules=[__import__("klix").rules.Rule(label="a", any_of=["c++", "c#"], mode="boost")]))
    eng.compile()
    eng.decide("problem in c++ build")  # must not raise, + must be escaped

def t_rule_force_multiple():
    """Two force rules matching: first must win deterministically."""
    R = __import__("klix").rules.Rule
    eng = DecisionEngine()
    eng.add_head(Choice(name="c", options={"a": ["alpha"], "b": ["beta"]},
                        rules=[R(label="a", any_of=["x"], mode="force"),
                               R(label="b", any_of=["x", "y"], mode="force")]))
    eng.compile()
    vals = {eng.decide("x and y").c for _ in range(3)}
    assert len(vals) == 1, f"ambiguous force: {vals}"

def t_rule_boost_none_value():
    """Boost on a rejected (None) result: must not crash or resurrect value."""
    R = __import__("klix").rules.Rule
    eng = DecisionEngine()
    eng.add_head(Choice(name="c", options={"a": ["alpha one two"], "b": ["beta one two"]},
                        reject_anchors=["smalltalk"],
                        rules=[R(label="a", any_of=["zzz"], mode="boost", weight=0.1)]))
    eng.compile()
    r = eng.decide("completely unrelated off-domain text zzz")
    # value may be None or 'a'; must not raise
    assert r.c is None or r.c == "a"

# ---------------------------------------------------------------- calibrate
def t_calibrate_all_same_label():
    eng = DecisionEngine()
    eng.add_head(Flag(name="f", true_anchors=["attack"], false_anchors=["printer"]))
    eng.compile()
    rep = eng.calibrate("f", [("a", True), ("b", True), ("c", True), ("d", True),
                              ("e", True), ("f2", True)])
    # degenerate: only one class -> must warn, not crash
    print(f"        all-true report: {rep}")

def t_calibrate_empty_samples():
    eng = DecisionEngine()
    eng.add_head(Flag(name="f", true_anchors=["attack"], false_anchors=["printer"]))
    eng.compile()
    try:
        eng.calibrate("f", [])
        issues.append(("calibrate empty samples", "no error raised"))
        print("  FAIL  calibrate empty: no error")
    except ValueError:
        print("  OK    calibrate empty raises")

def t_score_calibrate_conflicting_targets():
    eng = DecisionEngine()
    eng.add_head(Score(name="s", low_anchors=["calm"], high_anchors=["crisis"]))
    eng.compile()
    rep = eng.calibrate("s", [("calm text", 3.0), ("calm text", 0.0), ("calm text", 1.5)])
    print(f"        conflicting targets: a={rep['a']:.2f} b={rep['b']:.2f} (degenerate ok)")

# ---------------------------------------------------------------- validate
def t_validate_empty_options():
    eng = DecisionEngine()
    eng.add_head(Choice(name="c", options={"a": ["alpha one"], "b": ["beta one"]}))
    eng.compile()
    rep = eng.validate_anchors()
    assert isinstance(rep, list)

def t_validate_many_single_anchor_classes():
    eng = DecisionEngine()
    eng.add_head(Choice(name="c", options={f"k{i}": [f"text {i}"] for i in range(12)}))
    eng.compile()
    rep = eng.validate_anchors()  # pairwise on 12 classes -> 66 pairs, must be fast
    assert isinstance(rep, list)

# ---------------------------------------------------------------- explain
def t_explain_empty_query():
    eng = DecisionEngine()
    eng.add_head(Choice(name="c", options={"a": ["alpha one"], "b": ["beta one"]}))
    eng.compile()
    r = eng.decide("")
    exp = r.explain("c")  # must not raise even though no tokens
    print(f"        empty explain: {str(exp)[:80]}")

def t_explain_unknown_vocab():
    eng = DecisionEngine()
    eng.add_head(Choice(name="c", options={"a": ["alpha xyzq"], "b": ["beta xyzq"]}))
    eng.compile()
    r = eng.decide("ganz andere wörter hier")
    exp = r.explain("c")  # no token overlap -> must not raise
    print(f"        off-vocab explain keys: {sorted(exp.keys())}")

# ---------------------------------------------------------------- batch + heads
def t_batch_with_rules_and_reject():
    from klix.rules import Rule
    eng = DecisionEngine()
    eng.add_head(Choice(name="c", options={"a": ["alpha one"], "b": ["beta one"]},
                        reject_anchors=["small talk"],
                        rules=[Rule(label="a", any_of=["force"], mode="force")]))
    eng.compile()
    out = eng.decide_batch(["force this", "small talk", "alpha one"])
    vals = [r.c for r in out]
    assert vals[0] == "a", f"force rule not applied in batch: {vals}"
    print(f"        batch vals: {vals}")

def t_custom_head_zero_anchors_decide():
    class Empty(BaseHead):
        def __init__(self): super().__init__("e")
        def get_reference_texts(self): return []
        def fit(self, backbone): pass
        def evaluate(self, encoded): return {"value": None}
    eng = DecisionEngine()
    eng.add_head(Choice(name="c", options={"a": ["alpha one"], "b": ["beta one"]}))
    eng.add_head(Empty())
    eng.compile()
    r = eng.decide("alpha")
    exp = r.explain("e")  # default explain fallback
    assert "note" in exp

print("\n=== Bug hunt: edge cases ===")
for name, fn in [
    ("empty string decide", t_empty_string),
    ("whitespace-only decide", t_whitespace_only),
    ("unicode/emoji decide", t_unicode_emoji),
    ("determinism", t_determinism),
    ("batch mixed lengths", t_batch_mixed_lengths),
    ("batch 500 texts", t_batch_very_large),
    ("rule on empty text", t_rule_empty_text),
    ("rule case-insensitive", t_rule_case_insensitive),
    ("rule regex special chars", t_rule_regex_special_chars),
    ("two force rules deterministic", t_rule_force_multiple),
    ("boost on rejected text", t_rule_boost_none_value()),
]:
    check(name, fn)

def t_rule_boost_none_value():
    pass  # placeholder, real one defined above

check("calibrate all-same-label", t_calibrate_all_same_label)
check("calibrate empty samples", t_calibrate_empty_samples)
check("score calibrate conflicting", t_score_calibrate_conflicting_targets)
check("validate empty findings", t_validate_empty_options)
check("validate 12 single-anchor classes", t_validate_many_single_anchor_classes)
check("explain on empty query", t_explain_empty_query)
check("explain off-vocab query", t_explain_unknown_vocab)
check("batch with rules+reject", t_batch_with_rules_and_reject)
check("custom head explain fallback", t_custom_head_zero_anchors_decide if False else t_custom_head_zero_anchors_decide)

print("\n=== Summary ===")
if issues:
    print(f"{len(issues)} ISSUES:")
    for name, err in issues:
        print(f"  - {name}: {err}")
else:
    print("No issues found.")
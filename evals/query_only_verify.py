"""Verification: is query-side-only expansion really as good as anchor+query?

Counter-intuitive result from the diagnostic: dropping anchor-side expansion
entirely (variant C) fixed the monolingual regression WITHOUT losing the
cross-lingual gain. That deserves a wider test than one small set, because it
would simplify the design substantially (query-only is cheaper: no extra anchor
embeddings, no IDF shift).

Why it can work: the query is expanded with the foreign terms, and the anchors
that ALREADY contain those terms are found via the shared vocabulary. The
anchor's original wording is also still in the vocabulary, so same-language
matching is unaffected. Anchor expansion is then redundant — and harmful when
it shifts IDF on monolingual schemas.

Measured here on all three groups of the main benchmark, plus the full 60-case
frozen set to check for collateral damage.

Run: uv run python -m evals.query_only_verify
"""
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from klix import Choice, DecisionEngine, load_glossary  # noqa: E402
from evals.eval_domains import IMG_CASES, IMG_OPTIONS, SHOP_CASES, SHOP_OPTIONS, TASK_CASES, TASK_OPTIONS  # noqa: E402
from evals.glossary_bench import (  # noqa: E402
    DE_ANCHORS, DE_ONLY_ANCHORS, DE_ONLY_QUERIES, DE_QUERIES, EN_ANCHORS,
    EN_QUERIES,
)
from evals.variant_sweep import FIN_OPTIONS, FIN_TESTS, HR_OPTIONS, HR_TESTS  # noqa: E402

FROZEN = [
    ("HR", HR_OPTIONS, HR_TESTS),
    ("FIN", FIN_OPTIONS, FIN_TESTS),
    ("IMAGE", IMG_OPTIONS, [(q, e) for q, e in IMG_CASES if e]),
    ("TASK", TASK_OPTIONS, [(q, e) for q, e in TASK_CASES if e]),
    ("SHOP", SHOP_OPTIONS, [(q, e) for q, e in SHOP_CASES if e]),
]


class QueryOnlyGlossary(Choice):
    """Glossary on queries, anchors untouched (candidate for v0.8.8)."""

    def fit(self, backbone):
        saved = self.glossary
        self.glossary = None
        super().fit(backbone)
        self.glossary = saved


def acc(head_cls, anchors, queries, **kw):
    eng = DecisionEngine()
    eng.add_head(head_cls(name="r", options=anchors, **kw))
    eng.compile()
    return sum(1 for t, e in queries if eng.decide(t).r == e), len(queries)


def show(label, head_cls, anchors, queries, **kw):
    ok, n = acc(head_cls, anchors, queries, **kw)
    print(f"  {label:40s} {ok:3d}/{n} = {ok/n:6.1%}")
    return ok, n


GROUPS = [
    ("EN anchors <- DE queries", EN_ANCHORS, DE_QUERIES),
    ("DE anchors <- EN queries", DE_ANCHORS, EN_QUERIES),
    ("DE anchors <- DE queries (control)", DE_ONLY_ANCHORS, DE_ONLY_QUERIES),
]

print("=" * 92)
print("THREE GROUPS: v0.8.7 (anchor+query) vs query-only vs baseline")
print("=" * 92)
for gname, anchors, queries in GROUPS:
    print(f"\n-- {gname} --")
    b = show("baseline (no glossary)", Choice, anchors, queries)
    a = show("v0.8.7 anchor+query expansion", Choice, anchors, queries,
             glossary=load_glossary(), glossary_weight=0.4)
    c = show("query-only expansion", QueryOnlyGlossary, anchors, queries,
             glossary=load_glossary(), glossary_weight=0.4)
    print(f"     query-only vs baseline: {c[0]-b[0]:+d}   vs v0.8.7: {c[0]-a[0]:+d}")

print()
print("=" * 92)
print("FROZEN 60-CASE SET: collateral damage check")
print("=" * 92)
print("(these schemas are mostly monolingual; the glossary has little to add)")
tot = {"base": [0, 0], "a": [0, 0], "c": [0, 0]}
for name, opts, tests in FROZEN:
    b = acc(Choice, opts, tests)
    a = acc(Choice, opts, tests, glossary=load_glossary(), glossary_weight=0.4)
    c = acc(QueryOnlyGlossary, opts, tests, glossary=load_glossary(), glossary_weight=0.4)
    for key, val in (("base", b), ("a", a), ("c", c)):
        tot[key][0] += val[0]
        tot[key][1] += val[1]
    print(f"  {name:8s} base {b[0]:2d}/{b[1]}   v0.8.7 {a[0]:2d}/{a[1]}   query-only {c[0]:2d}/{c[1]}")
print()
print(f"  TOTAL  base {tot['base'][0]}/{tot['base'][1]}   "
      f"v0.8.7 {tot['a'][0]}/{tot['a'][1]}   "
      f"query-only {tot['c'][0]}/{tot['c'][1]}")
print()
print("Decision rule: ship query-only IF it keeps the cross-lingual gains AND")
print("does not lose on the frozen set. Otherwise keep v0.8.7 anchor+query.")

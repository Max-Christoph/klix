"""Run klix on JevBench's public `choice` tier — anchors taken from the source's own criteria.

WHY THIS FILE EXISTS, AND WHAT IT IS NOT
----------------------------------------
JevBench (github.com/fstandhartinger/jevbench, MIT) is a benchmark for **typed
decision models used for LLM routing**: state + a question + candidate labels, and
the system picks one. klix is a support/process text classifier. Running klix here
is a **repurposing**, not a like-for-like test of the kind of task klix is built for.
The number this script prints is not a verdict on klix, and it is deliberately NOT
comparable to the Tev1 / Nimble / Jev percentages that circulate publicly — different
subset, different methodology, and the source itself states there is no shared
System-One benchmark.

The anchors are **the source's own criteria strings** (`question.criteria.values()`),
one anchor per candidate label. They are not written by us. That choice matters:
inventing anchors would have measured our phrasing, not the mapping. Using the
source's own rubrics keeps the run to "can a keyword/embedding classifier recover
the label from the text alone", which is the honest question at this anchor count.

LICENCE (checked before use, per the project's data rule)
--------------------------------------------------------
`datasets/public/{original,easy,hard}.jsonl`, filtered to `question.type == "choice"`:
139 cases, **139/139 carry `provenance.license == "MIT"`** and `source` values of
"JevBench original authored scenario" / "JevBench v1.1 easy tier" / "JevBench v1.2
hard tier (original authored scenario)". Zero cases reference a third-party dataset.
The criteria dict is embedded in every case (139/139), so it travels inside the same
MIT-covered file rather than being pulled from an external corpus. The third-party
material JevBench does import (Massive / BANKING77 / etc.) is explicitly *not*
redistributed upstream — those live in split files this script never touches.

DATA IS NOT VENDORED HERE
-------------------------
Nothing from JevBench is copied into this repository. The script downloads the three
public JSONL files at run time from raw.githubusercontent.com. If they are not
reachable, it stops — it does not fall back to a cached copy, because a stale copy
would silently change what "139 cases" means.

METRIC
------
Top-1 exact match of the label, per classifier, with a bootstrap CI over cases.
That is the source's own headline metric for `choice` (accuracy over answered
cases). Per-class recall and the full misclassification list are printed because an
aggregate hides which labels fail; the error list is the primary output.

Run:
    uv run python -m evals.run_jevbench
    uv run python -m evals.run_jevbench --show-errors 0      # all errors
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from klix import Choice, DecisionEngine  # noqa: E402

RAW = "https://raw.githubusercontent.com/fstandhartinger/jevbench/main/datasets/public"
FILES = ("original.jsonl", "easy.jsonl", "hard.jsonl")
UA = "klix-jevbench-run/0.1"
SEED = 20260928
N_BOOT = 2000


def fetch(name: str) -> str:
    url = f"{RAW}/{name}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(
            f"ABORT: cannot fetch {url} ({exc}).\n"
            f"       JevBench data is deliberately NOT vendored in this repo, so a "
            f"run needs network access. No cached fallback is used on purpose.")


def render_state(state) -> str:
    """Flattens a case's `state` into text klix can classify.

    JevBench's `state` is a plain string in `original` and `easy`, but **18 of the 67
    `hard` choice cases carry a structured dict** (keys like `conversation`,
    `agent_policy`, `contacts_matching_*`). Passing that dict straight through raises
    `TypeError: TextEncodeInput must be Union[...]` inside the tokenizer — which is
    how this was found, not by reading the schema.

    The dict is serialized to a readable text form rather than dropped, because
    dropping would silently reduce n from 139 to 121 and the reported denominator
    would be wrong. The conversion is deliberately dumb (keys as headings, values in
    order) — no summarisation, no reordering, nothing that could flatter klix.
    """
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        parts = []
        for k, v in state.items():
            if isinstance(v, list):
                rendered = "\n".join(
                    json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x
                    for x in v)
            elif isinstance(v, str):
                rendered = v
            else:
                rendered = json.dumps(v, ensure_ascii=False)
            parts.append(f"{k}:\n{rendered}")
        return "\n\n".join(parts)
    return json.dumps(state, ensure_ascii=False)


def load_choice_cases():
    """Returns [{id, state, expected, labels, criteria, license, source, tier}]."""
    cases = []
    for name in FILES:
        body = fetch(name)
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            q = d.get("question") or {}
            if q.get("type") != "choice":
                continue
            prov = d.get("provenance") or {}
            raw_state = d.get("state")
            cases.append({
                "id": d.get("id"),
                "state": render_state(raw_state),
                "state_was_structured": not isinstance(raw_state, str),
                "expected": d.get("expected"),
                "labels": list(d.get("labels") or []),
                "criteria": dict(q.get("criteria") or {}),
                "license": prov.get("license"),
                "source": prov.get("source"),
                "tier": name.replace(".jsonl", ""),
            })
    return cases


def licence_gate(cases):
    """Refuses to run unless every case is MIT-covered and criterion-anchored.

    This is the check the project's data rule demands *before* use, not after. It
    fails loudly rather than filtering quietly, because a silently reduced case set
    would make the reported n meaningless.
    """
    problems = []
    for c in cases:
        if c["license"] != "MIT":
            problems.append(f"{c['id']}: license={c['license']!r}, expected 'MIT'")
        if not c["criteria"]:
            problems.append(f"{c['id']}: no embedded criteria (anchor source missing)")
        missing = [lab for lab in c["labels"] if lab not in c["criteria"]]
        if missing:
            problems.append(f"{c['id']}: labels without a criterion text: {missing}")
        if c["expected"] not in c["labels"]:
            problems.append(f"{c['id']}: expected {c['expected']!r} not among labels")
    # A third-party-sourced case would show up as a non-JevBench source string.
    foreign = [c["id"] for c in cases if "jevbench" not in (c["source"] or "").lower()]
    problems += [f"{i}: source is not a JevBench authored tier" for i in foreign]
    if problems:
        print("ABORT: licence / provenance gate failed, nothing run:")
        for p in problems[:20]:
            print("   ", p)
        if len(problems) > 20:
            print(f"    ... {len(problems) - 20} more")
        raise SystemExit(1)


def anchors_from_criteria(case):
    """One anchor per candidate label = the source's own criterion sentence."""
    return {lab: [case["criteria"][lab]] for lab in case["labels"]}


def bootstrap_ci(flags, n_boot: int = N_BOOT, seed: int = SEED):
    rng = random.Random(seed)
    n = len(flags)
    if n == 0:
        return 0.0, 0.0, 0.0
    means = sorted(sum(flags[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    return sum(flags) / n, means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def run(cases, classifier, shared_backbone=None):
    """One engine per case: labels and anchors differ from case to case.

    That is inherent to this dataset — the candidate set is per-task, not global —
    so compile() runs once per case. `DecisionEngine.__init__` always constructs a
    fresh `HybridBackbone` (a new ONNX model), which would mean 139 model loads per
    classifier. The engine accepts no backbone argument, so the loaded backbone is
    swapped in after construction instead: the encoder is stateless, and reusing it
    is what makes this run finish. Documented because it is a real trick, not an
    obvious one.
    """
    flags, preds, confs = [], [], []
    for c in cases:
        anchors = anchors_from_criteria(c)
        eng = DecisionEngine()
        if shared_backbone is not None:
            eng.backbone = shared_backbone
        eng.add_head(Choice(name="route", options=anchors, classifier=classifier))
        eng.compile()
        res = eng.decide(c["state"])
        got = getattr(res, "route", None)
        preds.append(got)
        try:
            detail = res.details("route")
            conf = detail.get("confidence") if isinstance(detail, dict) else None
        except Exception:  # noqa: BLE001 — explanation is best-effort
            conf = None
        confs.append(conf if isinstance(conf, (int, float)) else None)
        flags.append(1 if got == c["expected"] else 0)
    return flags, preds, confs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classifier", action="append", default=None,
                    help="repeatable; default nearest, centroid, linear")
    ap.add_argument("--show-errors", type=int, default=40,
                    help="misclassifications to print (0 = all)")
    args = ap.parse_args()
    classifiers = args.classifier or ["nearest", "centroid", "linear"]

    cases = load_choice_cases()
    licence_gate(cases)

    per_tier = {}
    for c in cases:
        per_tier[c["tier"]] = per_tier.get(c["tier"], 0) + 1

    print("=" * 100)
    print("klix on JevBench public `choice` tier — REPURPOSED RUN, read the caveats")
    print("=" * 100)
    print(f"  cases      : {len(cases)}  ({', '.join(f'{k} {v}' for k, v in sorted(per_tier.items()))})")
    n_struct = sum(1 for c in cases if c["state_was_structured"])
    print(f"  anchors    : the source's own criteria strings, one per candidate label")
    print(f"               (NOT written by us — so this measures the mapping, not our phrasing)")
    print(f"  labels     : per-case candidate set, {statistics.median(len(c['labels']) for c in cases):.0f} median")
    if n_struct:
        print(f"  state form : {len(cases) - n_struct} plain strings, {n_struct} serialized from a "
              f"structured dict")
        print(f"               (the dict is rendered to text, NOT dropped — dropping would")
        print(f"                silently shrink n and make the denominator wrong)")
    print(f"  licence    : MIT on every case (gated above, aborts otherwise)")
    print(f"  metric     : top-1 exact match, bootstrap CI {N_BOOT} resamples, seed {SEED}")
    print()
    print("  THIS IS NOT COMPARABLE to published Tev1 / Nimble / Jev percentages:")
    print("  different subset (public choice tier only), different methodology (no")
    print("  instruction model, no probabilities, per-case anchor sets), and the source")
    print("  itself states there is no shared System-One benchmark.")
    print()

    results = {}
    shared = None
    for clf in classifiers:
        if shared is None:
            # Built once and reused for every case AND every classifier — see `run`.
            from klix.backbone import HybridBackbone
            shared = HybridBackbone()
        flags, preds, confs = run(cases, clf, shared_backbone=shared)
        acc, lo, hi = bootstrap_ci(flags)
        results[clf] = (flags, preds, confs)
        print(f"# classifier = {clf!r}")
        print(f"  accuracy : {sum(flags)}/{len(flags)} = {acc:.1%}  [95% CI {lo:.1%}, {hi:.1%}]")
        # per-tier, because the three tiers are very different difficulty
        for tier in sorted(per_tier):
            idx = [i for i, c in enumerate(cases) if c["tier"] == tier]
            hit = sum(flags[i] for i in idx)
            print(f"    tier {tier:9} {hit:3}/{len(idx):<3} {hit/len(idx):6.1%}")
        print(f"  misclassified: {sum(1 for f in flags if not f)}/{len(flags)}")
        print()

    print("=" * 100)
    print("MISCLASSIFICATIONS — the primary output; the aggregate hides the pattern")
    print("=" * 100)
    for clf, (flags, preds, confs) in results.items():
        errs = [i for i, f in enumerate(flags) if not f]
        if not errs:
            print(f"\n# {clf!r}: none")
            continue
        shown = errs if args.show_errors == 0 else errs[:args.show_errors]
        print(f"\n# classifier = {clf!r}  ({len(errs)} errors)")
        for i in shown:
            c = cases[i]
            conf = confs[i]
            conf_s = f"{conf:.2f}" if isinstance(conf, float) else "  - "
            state = c["state"].replace("\n", " ")[:58]
            print(f"  [{c['tier']:6}] {str(c['expected'])[:26]:28} -> {str(preds[i])[:26]:28} "
                  f"conf={conf_s}  {state}")
        if len(shown) < len(errs):
            print(f"  ... {len(errs) - len(shown)} more (use --show-errors 0)")

    out = Path("evals") / "jevbench_result.json"
    out.write_text(json.dumps({
        "n": len(cases), "per_tier": per_tier, "seed": SEED, "n_boot": N_BOOT,
        "results": {clf: {"correct": sum(f), "n": len(f)} for clf, (f, _, _) in results.items()},
        "caveat": "repurposed run; anchors are the source's criteria strings; not "
                  "comparable to published Tev1/Nimble/Jev figures",
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

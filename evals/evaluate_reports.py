"""Evaluation of klix on real, uncurated reports — CSV in, numbers and errors out.

PURPOSE: evaluation only. No feature, no tuning, no training. It reads labelled
reports and reports what the engine does with them.

Input
-----
A CSV with a header row and at least two columns:

    text,expected
    "Der Server ist ausgefallen",technical
    "Rechnung stimmt nicht",billing

Delimiter and column names are configurable (`--text-col`, `--label-col`,
`--delimiter`). UTF-8 is assumed; a BOM and a semicolon delimiter are handled
automatically, because that is what Excel exports here.

Schema is supplied on the command line as a JSON file:

    {"billing": ["refund for my order is missing", ...],
     "technical": ["the server keeps crashing", ...]}

Optionally pass a glossary JSON (the same format `DecisionEngine(glossary=...)`
accepts) with `--glossary`. It is reported as a separate row, never applied
silently — the point is to see whether it helps on real data.

Output
------
1. Accuracy per classifier, with a bootstrap CI over the test cases.
2. Per-class recall and a confusion-style summary of the misclassifications.
3. The full list of misclassifications, to read through.

The misclassification list is the primary output, not an appendix: on real,
uncurated data the aggregate accuracy says little without knowing *which* reports
fail and how. It prints text, expected, predicted and the decision's confidence.

Run:
    uv run python -m evals.evaluate_reports --csv reports.csv --schema schema.json
    uv run python -m evals.evaluate_reports --csv reports.csv --schema schema.json \
        --glossary glossary.json --classifier centroid --classifier linear
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from klix import Choice, DecisionEngine  # noqa: E402

SEED = 20260928
N_BOOT = 2000


def read_csv(path: Path, delimiter: str | None, text_col: str, label_col: str):
    """Returns [(text, label)]. Tolerates BOM and a semicolon delimiter."""
    raw = path.read_text(encoding="utf-8-sig")
    if delimiter is None:
        # Sniff on the first line; Excel exports here use ';'.
        first = raw.splitlines()[0] if raw.splitlines() else ""
        delimiter = ";" if first.count(";") > first.count(",") else ","
    rows = list(csv.DictReader(raw.splitlines(), delimiter=delimiter))
    if not rows:
        raise SystemExit(f"no data rows in {path}")
    missing = [c for c in (text_col, label_col) if c not in rows[0]]
    if missing:
        raise SystemExit(
            f"column(s) {missing} not found in {path}; available: {list(rows[0])}")
    out = []
    for r in rows:
        text = (r.get(text_col) or "").strip()
        label = (r.get(label_col) or "").strip()
        if text and label:
            out.append((text, label))
    return out


def bootstrap_ci(flags, n_boot: int = N_BOOT, seed: int = SEED):
    """95 % CI of the accuracy, resampling the cases."""
    rng = random.Random(seed)
    n = len(flags)
    if n == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_boot):
        s = sum(flags[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    return sum(flags) / n, means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def evaluate(rows, schema, classifier, glossary=None):
    """Returns (flags, predictions, confidences). One engine per run.

    Confidence comes from `DecisionResult.details(head)[...]`, not from the result
    object itself — `DecisionResult` exposes `data`/`details`/`latency_ms`, and the
    per-head confidence lives in the details dict. It is reported because on real
    data "wrong" and "wrong but unsure" are different problems.
    """
    eng = DecisionEngine(glossary=glossary)
    eng.add_head(Choice(name="c", options=schema, classifier=classifier))
    eng.compile()
    flags, preds, confs = [], [], []
    for text, expected in rows:
        res = eng.decide(text)
        got = getattr(res, "c", None)
        preds.append(got)
        conf = None
        try:
            detail = res.details("c")
            if isinstance(detail, dict):
                conf = detail.get("confidence", detail.get("value"))
        except Exception:  # noqa: BLE001 — details are best-effort, never fatal
            conf = None
        confs.append(conf if isinstance(conf, (int, float)) else None)
        flags.append(1 if got == expected else 0)
    return flags, preds, confs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--schema", required=True, type=Path,
                    help="JSON: {label: [anchor, ...]}")
    ap.add_argument("--glossary", type=Path, default=None,
                    help="optional glossary JSON to evaluate alongside")
    ap.add_argument("--classifier", action="append", default=None,
                    help="repeatable; default centroid + linear")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="expected")
    ap.add_argument("--delimiter", default=None)
    ap.add_argument("--show-errors", type=int, default=50,
                    help="how many misclassifications to print (0 = all)")
    args = ap.parse_args()

    rows = read_csv(args.csv, args.delimiter, args.text_col, args.label_col)
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    classifiers = args.classifier or ["centroid", "linear"]

    glossary = None
    glossary_src = "none"
    if args.glossary is not None:
        from klix.glossary import resolve_glossary
        glossary = resolve_glossary(str(args.glossary))
        glossary_src = str(args.glossary)

    labels = sorted(schema)
    seen = {lab for _, lab in rows}
    unknown = sorted(seen - set(labels))
    if unknown:
        print(f"WARNING: {len(unknown)} label(s) in the CSV are not in the schema "
              f"and can never be predicted correctly: {unknown}")
        print(f"         those cases count as errors. Fix the schema or the CSV.\n")

    print("=" * 100)
    print("REPORT EVALUATION — klix on real, uncurated reports")
    print("=" * 100)
    print(f"  cases      : {len(rows)}")
    print(f"  labels     : {len(seen)} in data, {len(labels)} in schema")
    print(f"  schema     : {args.schema}")
    print(f"  glossary   : {glossary_src}")
    print(f"  bootstrap  : {N_BOOT} resamples, seed {SEED}")
    print()

    all_results = {}
    for classifier in classifiers:
        for use_glossary, tag in ((False, "no glossary"), (True, "with glossary")):
            if use_glossary and glossary is None:
                continue
            flags, preds, confs = evaluate(rows, schema, classifier,
                                          glossary if use_glossary else None)
            acc, lo, hi = bootstrap_ci(flags)
            all_results[(classifier, tag)] = (flags, preds, confs)
            print(f"# classifier = {classifier!r}, {tag}")
            print(f"  accuracy : {sum(flags)}/{len(flags)} = {acc:.1%}  "
                  f"[95% CI {lo:.1%}, {hi:.1%}]")

            # per-class recall — an aggregate hides which class carries the failures
            print(f"  recall per class:")
            for lab in labels:
                idx = [i for i, (_, e) in enumerate(rows) if e == lab]
                if not idx:
                    continue
                hit = sum(flags[i] for i in idx)
                bar = "#" * int(20 * hit / len(idx))
                print(f"    {lab:22} {hit:3}/{len(idx):<3} {hit/len(idx):6.1%}  {bar}")

            errs = [i for i, f in enumerate(flags) if not f]
            print(f"  misclassified: {len(errs)}/{len(rows)}")
            print()

    # ---- the primary output: read the failures ----
    if args.show_errors:
        print("=" * 100)
        print("MISCLASSIFICATIONS — read these; the aggregate hides the pattern")
        print("=" * 100)
        for (classifier, tag), (flags, preds, confs) in all_results.items():
            errs = [i for i, f in enumerate(flags) if not f]
            if not errs:
                continue
            shown = errs if args.show_errors == 0 else errs[:args.show_errors]
            print(f"\n# classifier = {classifier!r}, {tag}  ({len(errs)} errors)")
            for i in shown:
                text, expected = rows[i]
                conf = confs[i]
                conf_s = f"{conf:.2f}" if isinstance(conf, float) else "  - "
                print(f"  [{expected:>20} -> {str(preds[i]):<20}] conf={conf_s}  {text[:70]}")
            if len(shown) < len(errs):
                print(f"  ... {len(errs) - len(shown)} more (raise --show-errors)")

    # ---- machine-readable sidecar, so a run can be compared later ----
    out = {f"{clf}|{tag}": {
        "n": len(flags), "correct": sum(flags),
        "accuracy": (sum(flags) / len(flags)) if flags else 0.0,
    } for (clf, tag), (flags, _, _) in all_results.items()}
    side = args.csv.with_suffix(args.csv.suffix + ".eval.json")
    side.write_text(json.dumps({"cases": len(rows), "results": out},
                               indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

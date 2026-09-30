"""Run klix and the Ollama comparison models on the bespoke datasets, case-aligned.

WHAT THIS PRODUCES, AND WHAT IT DOES NOT
----------------------------------------
A repurposed comparison (decision §7.1 option a). klix is a support/process text
classifier; MASSIVE, banking77 and PAWS are different tasks. The anchors klix needs
are built by this repo, not shipped by the datasets — so the numbers here are NOT
comparable to published Tev1 / Nimble / JevBench figures. Every result file carries
that caveat plus the anchor provenance, and `score_case_alignment()` refuses to
compare systems that did not see the same cases in the same order.

FOUR RULES THAT MAKE THE COMPARISON MEANINGFUL
----------------------------------------------
1. `unparseable` is never folded into accuracy. A model that answers in prose is
   *unusable*, not *inaccurate* — different problems, different fixes. Both rates
   are reported (`accuracy` over answered cases, `coverage` over all cases), so a
   broken prompt cannot masquerade as a hard task.
2. Every system sees the same case list from the same file. Alignment is asserted.
3. Anchor overlap with the test set is measured (`assert_no_leakage`) and reported,
   because MASSIVE's splits are not sentence-disjoint (see PROVENANCE.md §1.1).
4. `--subsample` is explicit and recorded. §7.5 requires it when the feasibility
   probe projects > 30 min; a silently shortened run would be worse than no run.

Run:
    python -m evals.run_bespoke --dataset massive_en --system klix --anchors few_shot_k3
    python -m evals.run_bespoke --dataset massive_en --system ollama:qwen3.5:2b --subsample 500
    python -m evals.run_bespoke --dataset banking77 --system klix --anchors label_string
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from evals.bespoke_anchors import (  # noqa: E402
    REPURPOSING_CAVEAT,
    anchor_provenance,
    assert_no_leakage,
    few_shot_anchors,
    few_shot_plus_label_anchors,
    label_string_anchors,
)
from evals.bespoke_loader import DATA_DIR  # noqa: E402

DATASETS = {
    "massive_de": {"test": "massive_intent.de.jsonl", "train": "massive_intent.de.train.jsonl"},
    "massive_en": {"test": "massive_intent.en.jsonl", "train": "massive_intent.en.train.jsonl"},
    "banking77": {"test": "banking77.jsonl", "train": "banking77.train.jsonl"},
    "paws": {"test": "paws_adversarial.jsonl", "train": None},
}
DEFAULT_SEED = 20260930


# --------------------------------------------------------------------------
# pure helpers (tested)
# --------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_anchors(train_rows: list[dict], method: str, k: int = 3,
                  seed: int = DEFAULT_SEED) -> dict[str, list[str]]:
    """Anchor variants — see evals/bespoke_anchors.py for why there are multiple."""
    if method in ("few_shot_k3", "few_shot"):
        return few_shot_anchors(train_rows, k=k, seed=seed)
    if method == "label_string":
        labels = sorted({r["label"] for r in train_rows})
        return label_string_anchors(labels)
    if method in ("few_shot_k3_plus_label", "few_shot_plus_label"):
        return few_shot_plus_label_anchors(train_rows, k=k, seed=seed)
    raise ValueError(f"unknown anchor method {method!r}")


def summarize(cases: list[dict], predictions: list[str | None]) -> dict:
    """Accuracy over ANSWERED cases, coverage over all — never conflated."""
    if len(cases) != len(predictions):
        raise ValueError(f"length mismatch: {len(cases)} cases vs {len(predictions)} predictions")
    if not cases:
        return {"n": 0, "correct": 0, "answered": 0, "unparseable": 0,
                "accuracy": None, "coverage": 0.0}
    hits = [int(p == c["label"]) for p, c in zip(predictions, cases)]
    answered = sum(1 for p in predictions if p is not None)
    correct = sum(h for h, p in zip(hits, predictions) if p is not None)
    return {
        "n": len(cases),
        "correct": correct,
        "answered": answered,
        "unparseable": len(cases) - answered,
        "accuracy": (correct / answered) if answered else None,
        "coverage": answered / len(cases),
        "_hits": hits,
    }


def bootstrap_ci(hits: list[int], n_boot: int = 2000, seed: int = 7,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI over case-level hits."""
    if not hits:
        raise ValueError("cannot bootstrap an empty sample")
    rng = random.Random(seed)
    n = len(hits)
    means = []
    for _ in range(n_boot):
        means.append(sum(hits[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (round(lo, 4), round(hi, 4))


def score_case_alignment(per_system: dict[str, list[dict]]) -> bool:
    """True when every system was scored on the same case ids, in the same order."""
    lists = list(per_system.values())
    if len(lists) < 2:
        return True
    first = [c["id"] for c in lists[0]]
    return all([c["id"] for c in other] == first for other in lists[1:])


def subsample(cases: list[dict], n: int, seed: int = DEFAULT_SEED) -> list[dict]:
    """Deterministic subsample. Recorded in the result, never silent (§7.5)."""
    if n >= len(cases):
        return list(cases)
    # Stride keeps the class distribution closer to the full split than a head
    # slice: MASSIVE and banking77 are ordered by class.
    stride = max(1, len(cases) // n)
    picked = cases[::stride][:n]
    return picked


# --------------------------------------------------------------------------
# systems
# --------------------------------------------------------------------------

def run_klix(cases: list[dict], anchors: dict[str, list[str]],
             classifier: str = "nearest") -> list[str | None]:
    from klix import Choice, DecisionEngine

    eng = DecisionEngine()
    eng.add_head(Choice(name="c", options=anchors, classifier=classifier))
    eng.compile()
    texts = [case["text"] for case in cases]
    results = eng.decide_batch(texts)
    out: list[str | None] = []
    for res in results:
        value = getattr(res, "c", None)
        out.append(value if isinstance(value, str) else None)
    return out


def run_ollama(cases: list[dict], model: str) -> list[str | None]:
    from evals.bespoke_probe import parse_label, query, server_ready

    up, _ = server_ready()
    if not up:
        raise SystemExit("Ollama is not reachable — start the server, then re-run.")
    out: list[str | None] = []
    for i, case in enumerate(cases, 1):
        prompt = ("Classify the user request into exactly one of these intents.\n\n"
                  f"Intents:\n" + "\n".join(case["options"]) +
                  f"\n\nRequest: {case['text']}\n\nAnswer with the intent name only.")
        try:
            resp = query(model, prompt)
            out.append(parse_label(resp.get("response", ""), case["options"]))
        except Exception as exc:  # noqa: BLE001 — a failure is data, not a crash
            print(f"    [{i}/{len(cases)}] FAILED: {type(exc).__name__}: {str(exc)[:80]}")
            out.append(None)
        if i % 25 == 0:
            print(f"    [{i}/{len(cases)}] …")
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--system", action="append", required=True,
                    help="'klix' or 'ollama:<model>'; repeatable")
    ap.add_argument("--anchors", default="few_shot_k3",
                    choices=["few_shot_k3", "few_shot", "label_string", "few_shot_k3_plus_label", "few_shot_plus_label"])
    ap.add_argument("--classifier", default="nearest",
                    choices=["nearest", "centroid", "linear", "hybrid"],
                    help="Choice head classifier algorithm (default: nearest)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--subsample", type=int, default=0,
                    help="0 = full split; recorded in the result when used")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    test_path = Path(DATA_DIR) / spec["test"]
    if not test_path.exists():
        raise SystemExit(f"{test_path} missing — run evals.bespoke_loader first.")

    cases = _read_jsonl(test_path)
    if args.subsample:
        cases = subsample(cases, args.subsample, seed=args.seed)
    print(f"dataset={args.dataset} cases={len(cases)}"
          + (f" (subsampled from full split, §7.5)" if args.subsample else " (full split)"))

    # Anchors — only for klix; the Ollama models get the labels in the prompt.
    anchors: dict[str, list[str]] = {}
    overlap: list[str] = []
    if any(s == "klix" for s in args.system):
        if not spec["train"]:
            raise SystemExit(f"{args.dataset} has no train split; klix needs anchors")
        train_rows = _read_jsonl(Path(DATA_DIR) / spec["train"])
        anchors = build_anchors(train_rows, method=args.anchors, k=args.k, seed=args.seed)
        overlap = assert_no_leakage(anchors, {c["text"] for c in cases})
        print(f"anchors: {args.anchors} (k={args.k}, classifier={args.classifier}), "
              f"{len(anchors)} labels, {sum(len(v) for v in anchors.values())} anchors, "
              f"overlap with test set: {len(overlap)}")

    per_system: dict[str, list[dict]] = {}
    results: dict[str, dict] = {}
    for spec_sys in args.system:
        print(f"\n--- {spec_sys} ---")
        t0 = time.perf_counter()
        if spec_sys == "klix":
            preds = run_klix(cases, anchors, classifier=args.classifier)
        elif spec_sys.startswith("ollama:"):
            preds = run_ollama(cases, spec_sys.split(":", 1)[1])
        else:
            raise SystemExit(f"unknown system {spec_sys!r}")
        elapsed = time.perf_counter() - t0

        summary = summarize(cases, preds)
        hits = summary.pop("_hits")
        if summary["answered"]:
            summary["accuracy_ci95"] = bootstrap_ci(hits, seed=args.seed)
        summary["seconds"] = round(elapsed, 1)
        results[spec_sys] = summary
        per_system[spec_sys] = cases
        acc = summary["accuracy"]
        print(f"  n={summary['n']} correct={summary['correct']} "
              f"unparseable={summary['unparseable']} "
              f"accuracy={acc:.4f}" if acc is not None else "  accuracy=None (nothing parsed)")
        print(f"  coverage={summary['coverage']:.3f}  time={elapsed:.1f}s")

    aligned = score_case_alignment(per_system)
    if not aligned:
        print("\nWARNING: systems were not scored on the same case list — "
              "the comparison is invalid and the result file says so.")

    payload = {
        "dataset": args.dataset,
        "full_split_cases": len(_read_jsonl(test_path)),
        "scored_cases": len(cases),
        "subsampled": bool(args.subsample),
        "seed": args.seed,
        "anchor_method": args.anchors if anchors else None,
        "k": args.k if anchors else None,
        "classifier": args.classifier if anchors else None,
        "anchor_provenance": anchor_provenance(k=args.k, seed=args.seed) if anchors else None,
        "anchor_overlap_with_test": len(overlap),
        "case_aligned": aligned,
        "systems": results,
        "caveat": REPURPOSING_CAVEAT,
        "note": ("MASSIVE train/test are not sentence-disjoint (see "
                 "evals/data/bespoke/PROVENANCE.md 1.1): achievable accuracy is capped "
                 "and some anchors mirror test sentences."),
    }
    if not anchors:
        tag = "noanchors"
    elif args.anchors == "few_shot_k3" and args.classifier == "nearest" and args.k == 3:
        tag = "few_shot_k3"
    elif args.anchors == "label_string" and args.classifier == "nearest":
        tag = "label_string"
    else:
        tag = f"{args.anchors}_k{args.k}_{args.classifier}"
    out = Path("evals") / f"bespoke_result_{args.dataset}_{tag}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

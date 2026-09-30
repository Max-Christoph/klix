"""Run empirical multi-label evaluation on GoEmotions dataset.

Measures:
- Micro-Precision, Micro-Recall, Micro-F1
- Macro-Precision, Macro-Recall, Macro-F1
- Exact Match Ratio (Subset Accuracy)
- Hamming Loss
- Compile Time, Latency per doc, Bulk Throughput (docs/s)

Usage:
    uv run python -m evals.run_multilabel --k 3 --threshold 0.5 --subsample 500
    uv run python -m evals.run_multilabel --k 10 --threshold 0.5 --subsample 500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from klix import DecisionEngine, MultiLabel
from evals.multilabel_anchors import (
    anchor_provenance,
    assert_no_leakage,
    sample_multilabel_anchors,
)
from evals.multilabel_loader import EMOTION_NAMES, load_go_emotions_dataset


def compute_multilabel_metrics(
    y_true_list: list[list[str]],
    y_pred_list: list[list[str]],
    all_classes: list[str],
) -> dict:
    """Compute Micro/Macro F1, Subset Accuracy, and Hamming Loss."""
    n = len(y_true_list)
    if n == 0:
        return {}

    exact_matches = 0
    total_hamming_errors = 0
    total_binary_decisions = n * len(all_classes)

    class_tp = {c: 0 for c in all_classes}
    class_fp = {c: 0 for c in all_classes}
    class_fn = {c: 0 for c in all_classes}

    for y_true, y_pred in zip(y_true_list, y_pred_list):
        set_true = set(y_true)
        set_pred = set(y_pred)

        if set_true == set_pred:
            exact_matches += 1

        for c in all_classes:
            is_true = c in set_true
            is_pred = c in set_pred
            if is_true and is_pred:
                class_tp[c] += 1
            elif not is_true and is_pred:
                class_fp[c] += 1
                total_hamming_errors += 1
            elif is_true and not is_pred:
                class_fn[c] += 1
                total_hamming_errors += 1

    # Micro metrics
    sum_tp = sum(class_tp.values())
    sum_fp = sum(class_fp.values())
    sum_fn = sum(class_fn.values())

    micro_prec = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
    micro_rec = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
    micro_f1 = (
        (2 * micro_prec * micro_rec) / (micro_prec + micro_rec)
        if (micro_prec + micro_rec) > 0
        else 0.0
    )

    # Macro metrics (over classes that appear in ground truth or predictions)
    active_classes = [c for c in all_classes if (class_tp[c] + class_fn[c] + class_fp[c]) > 0]
    class_f1s = []
    class_precs = []
    class_recs = []

    for c in active_classes:
        tp = class_tp[c]
        fp = class_fp[c]
        fn = class_fn[c]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        class_precs.append(p)
        class_recs.append(r)
        class_f1s.append(f1)

    macro_prec = sum(class_precs) / len(class_precs) if class_precs else 0.0
    macro_rec = sum(class_recs) / len(class_recs) if class_recs else 0.0
    macro_f1 = sum(class_f1s) / len(class_f1s) if class_f1s else 0.0

    subset_accuracy = exact_matches / n
    hamming_loss = total_hamming_errors / total_binary_decisions

    return {
        "n_samples": n,
        "n_classes": len(all_classes),
        "micro_precision": round(micro_prec, 4),
        "micro_recall": round(micro_rec, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "exact_match_ratio": round(subset_accuracy, 4),
        "hamming_loss": round(hamming_loss, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Klix MultiLabel head on GoEmotions")
    parser.add_argument("--k", type=int, default=3, help="Anchors per class")
    parser.add_argument("--threshold", type=float, default=0.45, help="Decision threshold")
    parser.add_argument("--calibration", choices=["sigmoid", "linear", "cosine"], default="sigmoid")
    parser.add_argument("--sharpness", type=float, default=12.0)
    parser.add_argument("--center", type=float, default=0.38)
    parser.add_argument("--subsample", type=int, default=500, help="Evaluate on subsample (0 for full)")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    print(f"Loading GoEmotions dataset...")
    train_rows = load_go_emotions_dataset("train")
    test_rows = load_go_emotions_dataset("test")

    print(f"Loaded {len(train_rows)} train rows and {len(test_rows)} test rows.")

    # Sample anchors
    anchors = sample_multilabel_anchors(
        train_rows=train_rows,
        options=EMOTION_NAMES,
        k=args.k,
        seed=args.seed,
    )

    # Check leakage
    test_text_set = {r["text"].strip() for r in test_rows}
    leaked = assert_no_leakage(anchors, test_text_set, strict=False)
    print(f"Sampled {len(anchors)} categories x {args.k} anchors. Test leakage: {len(leaked)}")

    # Subsample test if requested
    if args.subsample > 0 and args.subsample < len(test_rows):
        step = len(test_rows) / args.subsample
        eval_cases = [test_rows[int(i * step)] for i in range(args.subsample)]
        is_subsampled = True
    else:
        eval_cases = test_rows
        is_subsampled = False

    print(f"Evaluating {len(eval_cases)} test cases (subsampled={is_subsampled})...")

    engine = DecisionEngine()
    engine.add_head(
        MultiLabel(
            name="emotions",
            options=anchors,
            threshold=args.threshold,
            calibration=args.calibration,
            sharpness=args.sharpness,
            center=args.center,
        )
    )

    t0_compile = time.perf_counter()
    engine.compile()
    compile_time = round(time.perf_counter() - t0_compile, 2)
    print(f"Engine compiled in {compile_time} s.")

    eval_texts = [r["text"] for r in eval_cases]
    y_true = [r["labels"] for r in eval_cases]

    t0_eval = time.perf_counter()
    results = engine.decide_batch(eval_texts)
    total_eval_time = time.perf_counter() - t0_eval

    y_pred = [res.emotions for res in results]

    metrics = compute_multilabel_metrics(y_true, y_pred, EMOTION_NAMES)
    latency_per_doc_ms = round((total_eval_time / len(eval_cases)) * 1000, 2)
    throughput_docs_per_s = round(len(eval_cases) / total_eval_time, 1)

    print("\n" + "=" * 60)
    print(f"GoEmotions MultiLabel Results (k={args.k}, thresh={args.threshold}, calib={args.calibration})")
    print("=" * 60)
    print(f"Micro-F1:            {metrics['micro_f1'] * 100:.2f} %")
    print(f"Micro-Precision:     {metrics['micro_precision'] * 100:.2f} %")
    print(f"Micro-Recall:        {metrics['micro_recall'] * 100:.2f} %")
    print(f"Macro-F1:            {metrics['macro_f1'] * 100:.2f} %")
    print(f"Exact Match (Subset):{metrics['exact_match_ratio'] * 100:.2f} %")
    print(f"Hamming Loss:        {metrics['hamming_loss']:.4f}")
    print(f"Compile Time:        {compile_time} s")
    print(f"Latency per doc:     {latency_per_doc_ms} ms/doc")
    print(f"Bulk Throughput:     {throughput_docs_per_s} docs/s")
    print("=" * 60)

    # Save artifact
    provenance = anchor_provenance(k=args.k, seed=args.seed)
    payload = {
        "dataset": "go_emotions",
        "n_classes": len(EMOTION_NAMES),
        "classes": EMOTION_NAMES,
        "n_evaluated": len(eval_cases),
        "subsampled": is_subsampled,
        "k": args.k,
        "threshold": args.threshold,
        "calibration": args.calibration,
        "sharpness": args.sharpness,
        "center": args.center,
        "compile_time_s": compile_time,
        "latency_ms_per_doc": latency_per_doc_ms,
        "throughput_docs_per_s": throughput_docs_per_s,
        "metrics": metrics,
        "provenance": provenance,
        "leaked_anchors_count": len(leaked),
    }

    out_path = args.out
    if not out_path:
        out_name = f"multilabel_result_go_emotions_k{args.k}_{args.calibration}.json"
        out_path = str(Path(__file__).parent / out_name)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved benchmark result to: {out_path}")


if __name__ == "__main__":
    main()

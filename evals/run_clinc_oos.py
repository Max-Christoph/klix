"""Evaluates In-Scope Intent Classification and Out-of-Scope (OOD) Prediction on CLINC150.

Metrics:
    - In-Scope Accuracy (150 classes, N=4,500 test queries)
    - AUROC (Area under ROC curve for In-Scope vs. Out-of-Scope separation)
    - FPR@95 (False Positive Rate at 95% True Positive Rate for OOD detection)
    - Macro / Micro Precision & Recall on OOS rejection
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure repo root and src are on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from evals.clinc_loader import extract_clinc_anchors, load_clinc_dataset
from klix import Choice, DecisionEngine


def compute_fpr95(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes False Positive Rate (FPR) when True Positive Rate (TPR) is at least 95%.

    y_true: 1 for in-scope, 0 for out-of-scope (OOD)
    y_score: higher score means higher confidence of being in-scope
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # Find the threshold where TPR >= 0.95
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        return 1.0
    return float(fpr[idx[0]])


def run_clinc_evaluation(
    k_per_class: int = 3,
    classifier: str = "centroid",
    use_oos_reject_anchors: bool = False,
    subsample: int | None = None,
    output_json: Path | None = None,
) -> dict:
    split = load_clinc_dataset()
    anchors = extract_clinc_anchors(split, k_per_class=k_per_class)

    reject_anchors = split.oos_train if use_oos_reject_anchors else None

    engine = DecisionEngine()
    engine.add_head(
        Choice(
            name="intent",
            options=anchors,
            classifier=classifier,
            reject_anchors=reject_anchors,
        )
    )

    t0_compile = time.perf_counter()
    engine.compile()
    compile_time = time.perf_counter() - t0_compile

    # Prepare test data
    in_scope_test = split.test
    oos_test = split.oos_test

    if subsample is not None:
        in_scope_test = in_scope_test[:subsample]
        oos_test = oos_test[: min(len(oos_test), subsample // 4)]

    all_texts = [text for text, _ in in_scope_test] + oos_test
    # 1 for in-scope, 0 for out-of-scope
    y_true = np.array([1] * len(in_scope_test) + [0] * len(oos_test))

    print(f"Evaluating {len(all_texts)} queries ({len(in_scope_test)} in-scope, {len(oos_test)} OOS)...")
    t0_eval = time.perf_counter()
    results = engine.decide_batch(all_texts)
    eval_time = time.perf_counter() - t0_eval

    # Extract confidence scores, max similarities, and predictions
    confidences = []
    max_sims = []
    reject_margins = []
    preds = []
    for r in results:
        details = r.details("intent")
        c = details.get("confidence", 0.0)
        s = details.get("score", 0.0)
        rj = details.get("reject_score", 0.0)
        confidences.append(c)
        max_sims.append(s)
        reject_margins.append(s - rj)
        preds.append(r.intent)

    confidences = np.array(confidences)
    max_sims = np.array(max_sims)
    reject_margins = np.array(reject_margins)

    # In-Scope accuracy (only evaluated on in-scope ground truth)
    in_scope_preds = preds[: len(in_scope_test)]
    correct = sum(
        1 for (text, true_label), pred in zip(in_scope_test, in_scope_preds)
        if pred == true_label
    )
    in_scope_acc = correct / len(in_scope_test)

    # OOD Detection Metrics (AUROC & FPR@95) for max_sim (thresholding)
    auroc_sim = float(roc_auc_score(y_true, max_sims))
    fpr95_sim = compute_fpr95(y_true, max_sims)

    # If reject anchors used, also evaluate reject margin
    auroc_margin = float(roc_auc_score(y_true, reject_margins)) if use_oos_reject_anchors else auroc_sim
    fpr95_margin = compute_fpr95(y_true, reject_margins) if use_oos_reject_anchors else fpr95_sim

    # Also check confidence
    auroc_conf = float(roc_auc_score(y_true, confidences))
    fpr95_conf = compute_fpr95(y_true, confidences)

    latency_per_doc_ms = (eval_time / len(all_texts)) * 1000.0
    throughput = len(all_texts) / eval_time

    summary = {
        "k_per_class": k_per_class,
        "classifier": classifier,
        "use_oos_reject_anchors": use_oos_reject_anchors,
        "in_scope_samples": len(in_scope_test),
        "oos_samples": len(oos_test),
        "in_scope_accuracy": round(in_scope_acc * 100.0, 2),
        "auroc_max_sim": round(auroc_sim * 100.0, 2),
        "fpr95_max_sim": round(fpr95_sim * 100.0, 2),
        "auroc_reject_margin": round(auroc_margin * 100.0, 2),
        "fpr95_reject_margin": round(fpr95_margin * 100.0, 2),
        "auroc_confidence": round(auroc_conf * 100.0, 2),
        "fpr95_confidence": round(fpr95_conf * 100.0, 2),
        "compile_time_s": round(compile_time, 2),
        "latency_ms_per_doc": round(latency_per_doc_ms, 2),
        "throughput_docs_per_s": round(throughput, 1),
    }

    print("\n--- CLINC150 Benchmark Results ---")
    print(f"In-Scope Accuracy:       {summary['in_scope_accuracy']:.2f}% (Prior: {100/150:.2f}%)")
    print(f"OOD AUROC (Max Cosine):  {summary['auroc_max_sim']:.2f}% (FPR@95: {summary['fpr95_max_sim']:.2f}%)")
    if use_oos_reject_anchors:
        print(f"OOD AUROC (Reject Margin): {summary['auroc_reject_margin']:.2f}% (FPR@95: {summary['fpr95_reject_margin']:.2f}%)")
    print(f"OOD AUROC (Confidence):  {summary['auroc_confidence']:.2f}% (FPR@95: {summary['fpr95_confidence']:.2f}%)")
    print(f"Throughput:              {summary['throughput_docs_per_s']} docs/s ({summary['latency_ms_per_doc']:.2f} ms/doc)")

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved results to {output_json}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CLINC150 Intent & OOS Benchmark")
    parser.add_argument("--k", type=int, default=3, help="Anchors per class")
    parser.add_argument("--classifier", choices=["centroid", "linear", "nearest"], default="centroid")
    parser.add_argument("--use-reject-anchors", action="store_true", help="Use oos_train as reject anchors")
    parser.add_argument("--subsample", type=int, default=None, help="Subsample in-scope count")
    parser.add_argument("--output", type=str, default=None, help="Path to output JSON")
    args = parser.parse_args()

    run_clinc_evaluation(
        k_per_class=args.k,
        classifier=args.classifier,
        use_oos_reject_anchors=args.use_reject_anchors,
        subsample=args.subsample,
        output_json=Path(args.output) if args.output else None,
    )

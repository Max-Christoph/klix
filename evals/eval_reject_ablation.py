"""Ablation: Reject Anchors vs. Thresholding at matched operating points on CLINC150 using DecisionEngine."""
import sys
from pathlib import Path

# Ensure repo root and src are on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import json
import numpy as np
from evals.clinc_loader import extract_clinc_anchors, load_clinc_dataset
from klix import Choice, DecisionEngine

def main():
    print("Loading CLINC150 dataset...")
    split = load_clinc_dataset()
    anchors = extract_clinc_anchors(split, k_per_class=10)
    oos_train_anchors = split.oos_train[:100]

    in_scope_test = split.test  # 4500
    oos_test = split.oos_test   # 1000

    print("Building engine WITH reject anchors...")
    engine_with_ra = DecisionEngine()
    engine_with_ra.add_head(
        Choice(
            name="intent",
            options=anchors,
            classifier="centroid",
            reject_anchors=oos_train_anchors,
        )
    )
    engine_with_ra.compile()

    print("Evaluating 4,500 in-scope test queries...")
    in_texts = [text for text, _ in in_scope_test]
    in_results_ra = engine_with_ra.decide_batch(in_texts)

    print("Evaluating 1,000 OOS test queries...")
    oos_results_ra = engine_with_ra.decide_batch(oos_test)

    # Extract scores
    in_sims = np.array([r.details("intent")["score"] for r in in_results_ra])
    in_reject_sims = np.array([r.details("intent")["reject_score"] for r in in_results_ra])
    in_is_rejected_ra = np.array([r.intent is None for r in in_results_ra])

    oos_sims = np.array([r.details("intent")["score"] for r in oos_results_ra])
    oos_reject_sims = np.array([r.details("intent")["reject_score"] for r in oos_results_ra])
    oos_is_rejected_ra = np.array([r.intent is None for r in oos_results_ra])

    # 1. Pure Reject Anchors (at tau=0.0)
    in_loss_ra_cnt = int(np.sum(in_is_rejected_ra))
    in_loss_ra_pct = (in_loss_ra_cnt / len(in_texts)) * 100.0

    oos_blocked_ra_cnt = int(np.sum(oos_is_rejected_ra))
    oos_blocked_ra_pct = (oos_blocked_ra_cnt / len(oos_test)) * 100.0

    print("\n=======================================================")
    print("1. PURE REJECT ANCHORS (tau = 0.0):")
    print(f"   In-Scope False Rejection: {in_loss_ra_cnt}/4500 ({in_loss_ra_pct:.2f}%)")
    print(f"   OOS Blocked:              {oos_blocked_ra_cnt}/1000 ({oos_blocked_ra_pct:.2f}%)")

    # 2. Pure Thresholding calibrated to match the EXACT same in-scope loss
    tau_matched = float(np.sort(in_sims)[in_loss_ra_cnt])
    in_rejected_thresh = in_sims < tau_matched
    oos_rejected_thresh = oos_sims < tau_matched

    oos_blocked_thresh_cnt = int(np.sum(oos_rejected_thresh))
    oos_blocked_thresh_pct = (oos_blocked_thresh_cnt / len(oos_test)) * 100.0

    print(f"\n2. PURE THRESHOLDING (Matched to {in_loss_ra_pct:.2f}% In-Scope Loss, tau = {tau_matched:.4f}):")
    print(f"   In-Scope False Rejection: {int(np.sum(in_rejected_thresh))}/4500 ({(np.sum(in_rejected_thresh)/len(in_texts))*100:.2f}%)")
    print(f"   OOS Blocked:              {oos_blocked_thresh_cnt}/1000 ({oos_blocked_thresh_pct:.2f}%)")

    # 3. Sweep across standard matched operating points: 5%, 10%, 15%, 20% In-Scope Loss
    print("\n=======================================================")
    print("MATCHED OPERATING POINT COMPARISON:")
    print("Loss % | Pure Thresholding OOS Blocked | Pure RA OOS Blocked | Combined (RA + Thresh) OOS Blocked")
    print("-----------------------------------------------------------------------------------------")

    matched_results = []
    for loss_pct in [round(in_loss_ra_pct, 2), 5.0, 10.0, 15.0, 20.0]:
        n_loss = int((loss_pct / 100.0) * len(in_texts))

        # A) Pure Thresholding
        tau_pure = float(np.sort(in_sims)[n_loss])
        oos_blk_pure = int(np.sum(oos_sims < tau_pure))

        # B) Combined (RA + threshold tau_comb such that total in-scope loss == n_loss)
        # Condition for in-scope rejected: in_is_rejected_ra | (in_sims < tau_comb)
        if n_loss <= in_loss_ra_cnt:
            # RA alone already rejects more or equal
            tau_comb = 0.0
            oos_blk_comb = oos_blocked_ra_cnt
        else:
            remaining_sims = in_sims[~in_is_rejected_ra]
            needed = n_loss - in_loss_ra_cnt
            tau_comb = float(np.sort(remaining_sims)[needed])
            in_comb_rej = in_is_rejected_ra | (in_sims < tau_comb)
            oos_comb_rej = oos_is_rejected_ra | (oos_sims < tau_comb)
            oos_blk_comb = int(np.sum(oos_comb_rej))

        print(f"{loss_pct:5.2f}% | {oos_blk_pure:4d}/1000 ({oos_blk_pure/10.0:5.1f}%)        | {oos_blocked_ra_cnt:4d}/1000 ({oos_blocked_ra_pct:5.1f}%)     | {oos_blk_comb:4d}/1000 ({oos_blk_comb/10.0:5.1f}%) [tau={tau_comb:.3f}]")

        matched_results.append({
            "in_scope_loss_pct": loss_pct,
            "pure_threshold_oos_blocked_pct": oos_blk_pure / 10.0,
            "pure_ra_oos_blocked_pct": oos_blocked_ra_pct,
            "combined_oos_blocked_pct": oos_blk_comb / 10.0,
            "tau_pure": tau_pure,
            "tau_comb": tau_comb,
        })

    with open("evals/ablation_reject_matched.json", "w") as f:
        json.dump(matched_results, f, indent=2)
    print("\nResults saved to evals/ablation_reject_matched.json")

if __name__ == "__main__":
    main()

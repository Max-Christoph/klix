"""Anchor quality validation: pairwise class-overlap report for Choice heads.

Purely read-only: this module NEVER mutates anchors. It computes:

- pairwise centroid cosine between option labels (are two classes one cloud?)
- shared / exclusive terms per pair (the actual confusers + sharpening hints)
- misplaced anchors (closer to a foreign centroid than to their own)
- structural warnings (single-anchor classes, duplicate anchors)

Design principle "honest signals": the report describes problems and suggests
fixes; applying or rejecting suggestions stays with the developer.
"""

import numpy as np

# Severity thresholds (cosine between class centroids). Measured on MiniLM:
# semantically distinct classes typically land at 0.2-0.45; partially
# overlapping wording (shared "approve"/"beantragen" patterns) lands 0.55-0.75.
MEDIUM_SEVERITY = 0.55
HIGH_SEVERITY = 0.70


def _centroid(vectors: np.ndarray) -> np.ndarray:
    """L2-normalized mean embedding of a (n x d) anchor-vector slice."""
    c = vectors.mean(axis=0)
    n = float(np.linalg.norm(c))
    return c / (n if n > 0 else 1.0)


def validate_choice_head(head) -> list[dict]:
    """Analyzes one compiled Choice head; returns a list of findings."""
    findings: list[dict] = []

    # --- Structural warnings ---------------------------------------------
    for label, rows in head._label_rows.items():
        texts = [head.flat_texts[r] for r in rows]
        if len(texts) == 1:
            findings.append({
                "head": head.name, "kind": "structure", "severity": "medium",
                "message": f"class '{label}' has only 1 anchor; add 2-3 more for stable routing",
            })
        seen: set[str] = set()
        for t in texts:
            key = t.strip().lower()
            if key in seen:
                findings.append({
                    "head": head.name, "kind": "structure", "severity": "medium",
                    "message": f"class '{label}' contains duplicate anchor {t!r}",
                })
            seen.add(key)

    # --- Pairwise centroid analysis ---------------------------------------
    centroids = {
        label: _centroid(head.dense_matrix[rows])
        for label, rows in head._label_rows.items()
    }
    labels = sorted(centroids.keys())
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            cos = float(np.dot(centroids[a], centroids[b]))
            if cos < MEDIUM_SEVERITY:
                continue  # well separated -> not interesting
            shared, excl_a, excl_b = _term_overlap(head, a, b)
            misplaced = _misplaced_anchors(head, a, b, centroids[a], centroids[b])
            severity = "high" if cos >= HIGH_SEVERITY else "medium"
            findings.append({
                "head": head.name, "kind": "overlap",
                "a": a, "b": b,
                "centroid_cos": round(cos, 3),
                "shared_terms": shared[:8],
                "a_exclusive": excl_a[:5],
                "b_exclusive": excl_b[:5],
                "misplaced": misplaced,
                "severity": severity,
            })

    return findings


def _term_overlap(head, a: str, b: str) -> tuple[list[str], list[str], list[str]]:
    """Tokens appearing in BOTH classes (TF-IDF-weighted), plus per-side exclusives."""
    def weighted_tokens(label: str) -> dict[str, float]:
        tokens: dict[str, float] = {}
        for r in head._label_rows[label]:
            row = head.sparse_matrix[r].tocsr()
            for col, val in zip(row.indices.tolist(), row.data.tolist()):
                token = head._token_for_col(col)
                if token:
                    tokens[token] = max(tokens.get(token, 0.0), float(val))
        return tokens

    ta, tb = weighted_tokens(a), weighted_tokens(b)
    shared = sorted((t for t in ta if t in tb), key=lambda t: -(ta[t] + tb.get(t, 0.0)))
    excl_a = sorted(set(ta) - set(tb), key=lambda t: -ta[t])
    excl_b = sorted(set(tb) - set(ta), key=lambda t: -tb[t])
    return shared, excl_a, excl_b


def _misplaced_anchors(head, a: str, b: str, cent_a: np.ndarray, cent_b: np.ndarray) -> list[dict]:
    """Anchors sitting closer to the other class's centroid than to their own
    (or within 5% of it) — likely mis-filed examples."""
    out: list[dict] = []
    for row in head._label_rows[a]:
        own = float(np.dot(head.dense_matrix[row], cent_a))
        other = float(np.dot(head.dense_matrix[row], cent_b))
        if other > own * 0.95:
            out.append({"text": head.flat_texts[row], "from": a, "closer_to": b})
    for row in head._label_rows[b]:
        own = float(np.dot(head.dense_matrix[row], cent_b))
        other = float(np.dot(head.dense_matrix[row], cent_a))
        if other > own * 0.95:
            out.append({"text": head.flat_texts[row], "from": b, "closer_to": a})
    return out
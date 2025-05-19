"""Score unsupervised detectors against the (held-out-for-eval) weak labels.

Anomaly detectors are unsupervised, but we still measure how well their scores
rank true attacks using PR-AUC / ROC-AUC (threshold-free) plus the best
achievable F1 and precision/recall at a chosen contamination threshold.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, contamination: float = 0.43) -> dict:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    pr_auc = float(average_precision_score(y_true, scores)) if y_true.any() else 0.0
    try:
        roc = float(roc_auc_score(y_true, scores))
    except ValueError:
        roc = 0.0

    # threshold at the top-`contamination` fraction of scores
    thr = np.quantile(scores, 1 - contamination)
    pred = (scores >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_at_thr = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # best achievable F1 over the PR curve
    prec, rec, _ = precision_recall_curve(y_true, scores)
    f1s = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    best_f1 = float(np.max(f1s)) if len(f1s) else 0.0

    return {
        "pr_auc": round(pr_auc, 3),
        "roc_auc": round(roc, 3),
        "precision@thr": round(precision, 3),
        "recall@thr": round(recall, 3),
        "f1@thr": round(f1_at_thr, 3),
        "best_f1": round(best_f1, 3),
        "threshold": round(float(thr), 4),
    }


def comparison_table(results: dict[str, dict]) -> str:
    cols = ["pr_auc", "roc_auc", "precision@thr", "recall@thr", "f1@thr", "best_f1"]
    header = f"{'model':<18} " + " ".join(f"{c:>13}" for c in cols)
    lines = [header, "-" * len(header)]
    for name, m in sorted(results.items(), key=lambda kv: kv[1].get("pr_auc", 0), reverse=True):
        lines.append(f"{name:<18} " + " ".join(f"{m.get(c, 0):>13}" for c in cols))
    return "\n".join(lines)

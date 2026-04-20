"""
Evaluation metrics for multi-label chest X-ray classification.

Implements:
  - per-class AUC-ROC
  - macro / weighted AUC
  - per-class F1, precision, recall (at threshold 0.5)
  - full evaluation report with pretty-printing
"""

import numpy as np
import json
import os
from typing import Optional

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    classification_report,
)


# ─── VinDr-CXR pathology names (labels 0-13, excludes "No finding") ──────────
PATHOLOGY_NAMES = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
]


def compute_auc_per_class(
    y_true: np.ndarray,     # (N, C) binary
    y_prob: np.ndarray,     # (N, C) sigmoid probabilities
) -> dict[str, float]:
    """Compute per-class AUC-ROC. Returns {class_name: auc_score}."""
    n_classes = y_true.shape[1]
    aucs = {}
    for c in range(n_classes):
        name = PATHOLOGY_NAMES[c] if c < len(PATHOLOGY_NAMES) else f"class_{c}"
        yt = y_true[:, c]
        yp = y_prob[:, c]
        if yt.sum() == 0 or yt.sum() == len(yt):
            # only one class present – AUC undefined
            aucs[name] = float("nan")
        else:
            aucs[name] = roc_auc_score(yt, yp)
    return aucs


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    class_names: Optional[list[str]] = None,
) -> dict:
    """
    Comprehensive multi-label evaluation.

    Returns a dict with:
      per_class_auc   : {name: float}
      macro_auc       : float
      weighted_auc    : float
      per_class_ap    : {name: float}  (average precision / PR-AUC)
      macro_f1        : float
      macro_precision : float
      macro_recall    : float
      per_class_f1    : {name: float}
    """
    n_classes = y_true.shape[1]
    names     = class_names or PATHOLOGY_NAMES[:n_classes]

    y_pred = (y_prob >= threshold).astype(int)

    # ── AUC ───────────────────────────────────────────────────────────────────
    per_class_auc = {}
    per_class_ap  = {}
    valid_aucs    = []
    support       = []

    for c in range(n_classes):
        name = names[c] if c < len(names) else f"class_{c}"
        yt   = y_true[:, c]
        yp   = y_prob[:, c]
        supp = int(yt.sum())
        support.append(supp)

        if supp == 0 or supp == len(yt):
            per_class_auc[name] = float("nan")
            per_class_ap[name]  = float("nan")
        else:
            auc = roc_auc_score(yt, yp)
            ap  = average_precision_score(yt, yp)
            per_class_auc[name] = auc
            per_class_ap[name]  = ap
            valid_aucs.append((auc, supp))

    # macro AUC (unweighted over valid classes)
    macro_auc = float(np.mean([v[0] for v in valid_aucs])) if valid_aucs else float("nan")

    # weighted AUC (weighted by support)
    total_support = sum(v[1] for v in valid_aucs)
    weighted_auc  = float(
        sum(v[0] * v[1] for v in valid_aucs) / total_support
    ) if total_support > 0 else float("nan")

    # ── F1 / Precision / Recall ───────────────────────────────────────────────
    macro_f1  = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    macro_p   = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r   = float(recall_score(y_true, y_pred, average="macro",   zero_division=0))

    per_class_f1 = {}
    f1_arr = f1_score(y_true, y_pred, average=None, zero_division=0)
    for c in range(n_classes):
        name = names[c] if c < len(names) else f"class_{c}"
        per_class_f1[name] = float(f1_arr[c])

    return {
        "per_class_auc":   per_class_auc,
        "macro_auc":       macro_auc,
        "weighted_auc":    weighted_auc,
        "per_class_ap":    per_class_ap,
        "macro_f1":        macro_f1,
        "macro_precision": macro_p,
        "macro_recall":    macro_r,
        "per_class_f1":    per_class_f1,
    }


def print_metrics(metrics: dict, title: str = "Evaluation Results") -> None:
    """Pretty-print evaluation metrics to stdout."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)

    print(f"  Macro AUC-ROC  : {metrics['macro_auc']:.4f}")
    print(f"  Weighted AUC   : {metrics['weighted_auc']:.4f}")
    print(f"  Macro F1       : {metrics['macro_f1']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall   : {metrics['macro_recall']:.4f}")
    print()

    print("  Per-class AUC-ROC:")
    for name, val in metrics["per_class_auc"].items():
        bar = "█" * int(val * 20) if not np.isnan(val) else "  (undefined)"
        score_str = f"{val:.4f}" if not np.isnan(val) else "  N/A "
        print(f"    {name:<28}  {score_str}  {bar}")

    print()
    print("  Per-class F1:")
    for name, val in metrics["per_class_f1"].items():
        bar = "█" * int(val * 20)
        print(f"    {name:<28}  {val:.4f}  {bar}")

    print(sep)


def save_metrics(metrics: dict, save_path: str, title: str = "") -> None:
    """Save metrics dict as a formatted JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    payload = {"title": title, **metrics}
    # Convert nan to null for JSON
    def _fix(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: _fix(v) for k, v in obj.items()}
        return obj

    with open(save_path, "w") as f:
        json.dump(_fix(payload), f, indent=2)
    print(f"[metrics] Saved → {save_path}")

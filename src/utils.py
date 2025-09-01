# src/utils.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def compute_threshold_curve(y_true: np.ndarray, proba: np.ndarray, step: float = 0.01) -> pd.DataFrame:
    """Return precision/recall/F1 across thresholds in [0,1]."""
    thresholds = np.arange(0.0, 1.0 + 1e-9, step)
    rows = []
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        rows.append(dict(threshold=t, precision=p, recall=r, f1=f1, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)))
    return pd.DataFrame(rows)


def best_f1_threshold(curve: pd.DataFrame) -> float:
    """Return threshold with max F1 (ties -> highest recall, then lowest threshold)."""
    top = curve.sort_values(["f1", "recall", "threshold"], ascending=[False, False, True]).head(1)
    return float(top["threshold"].iloc[0])


def threshold_for_precision_floor(curve: pd.DataFrame, floor: float) -> float | None:
    """Smallest threshold with precision >= floor (pick one with highest recall among them)."""
    sub = curve[curve["precision"] >= floor]
    if sub.empty:
        return None
    sub = sub.sort_values(["recall", "threshold"], ascending=[False, True])
    # choose the smallest threshold among the top recall rows
    best_recall = sub["recall"].iloc[0]
    candidates = sub[sub["recall"] == best_recall].sort_values("threshold")
    return float(candidates["threshold"].iloc[0])


def confusion_at_threshold(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    y_pred = (proba >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return dict(threshold=float(threshold), precision=float(p), recall=float(r), f1=float(f1),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

# src/train.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import yaml
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import load_and_clean
from src.preprocess import build_preprocessor

from src.utils import compute_threshold_curve, best_f1_threshold, threshold_for_precision_floor
import numpy as np
import pandas as pd

def main(config_path: str) -> None:
    # ---- Load config (robust to missing blocks) ----
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    data_cfg = cfg.get("data") or {}
    split_cfg = cfg.get("split") or {}
    model_cfg = cfg.get("model") or {}
    out_cfg = cfg.get("output") or {}

    # ---- Load data (Telco via kagglehub if csv_path is null/absent) ----
    X, y = load_and_clean(
        csv_path=data_cfg.get("csv_path"),
        target=data_cfg.get("target", "Churn"),
        positive_label=data_cfg.get("positive_label", "Yes"),
    )

    # Sanity prints before split
    print("Class counts (y, before split):")
    print(y.value_counts(dropna=False))
    if y.nunique() < 2:
        raise ValueError(f"Need at least 2 classes. Got counts: {y.value_counts().to_dict()}")

    # ---- Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_cfg.get("test_size", 0.2),
        random_state=cfg.get("seed", 42),
        stratify=y if split_cfg.get("stratify", True) else None,
    )
    print("Train class counts:", y_train.value_counts().to_dict())
    print("Test  class counts:", y_test.value_counts().to_dict())

    # ---- Preprocess ----
    preprocessor, _, _ = build_preprocessor(X_train)

    # ---- Model selection ----
    name = model_cfg.get("name", "logreg").lower()
    params = (model_cfg.get("params") or {}).copy()

    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        # sensible defaults if not provided
        params.setdefault("C", 1.0)
        params.setdefault("max_iter", 1000)
        params.setdefault("class_weight", "balanced")
        clf = LogisticRegression(**params)

    elif name == "xgb":
        from xgboost import XGBClassifier
        # handle imbalance with scale_pos_weight = N_neg / N_pos
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        spw = max(neg / max(pos, 1), 1.0)

        default_params = dict(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=1.0,
            objective="binary:logistic",
            tree_method="hist",
            n_jobs=-1,
            scale_pos_weight=spw,
            random_state=cfg.get("seed", 42),
            # Note: no early_stopping_rounds in xgboost>=3 sklearn API
        )
        default_params.update(params)
        clf = XGBClassifier(**default_params)

    else:
        raise ValueError(f"Unknown model: {name} (supported: logreg, xgb)")

    # ---- Pipeline & Train ----
    pipe = Pipeline([("preprocess", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)

    # --- Save feature names & background for SHAP ---
    feat_path = Path(out_cfg.get("feature_names_path", "models/feature_names.json"))
    bg_path   = Path(out_cfg.get("background_path", "models/background.joblib"))
    meta_path = Path(out_cfg.get("meta_path", "models/meta.json"))

    # Try to get feature names safely
    feature_names = None
    try:
        feature_names = pipe.named_steps["preprocess"].get_feature_names_out().tolist()
    except Exception:
        # Fallback: build from raw columns if needed
        feature_names = list(X_train.columns)

    # Small background sample for KernelExplainer fallback & for charts
    import numpy as np
    bg_idx = np.random.RandomState(cfg.get("seed", 42)).choice(
        len(X_train), size=min(200, len(X_train)), replace=False
    )
    X_bg = pipe.named_steps["preprocess"].fit_transform(X_train)  # already fitted above
    X_bg_small = X_bg[bg_idx]

    feat_path.parent.mkdir(parents=True, exist_ok=True)
    bg_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    import json, joblib
    json.dump(feature_names, open(feat_path, "w"))
    joblib.dump(X_bg_small, bg_path)

    # Minimal metadata for UI logic
    model_name = model_cfg.get("name", "logreg").lower()
    json.dump({"model": model_name}, open(meta_path, "w"))


    # ---- Evaluate ----
    proba = pipe.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    report = classification_report(y_test, (proba >= 0.5).astype(int), output_dict=True)

    # ---- Threshold curve & recommendations ----
    curve = compute_threshold_curve(y_test.to_numpy(), proba)
    best_f1 = best_f1_threshold(curve)
    p60 = threshold_for_precision_floor(curve, 0.60)
    p70 = threshold_for_precision_floor(curve, 0.70)

    # Augment metrics
    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "report": report,
        "thresholds": {
            "best_f1": best_f1,
            "precision>=0.60": p60,
            "precision>=0.70": p70,
        }
    }

    # Save validation predictions & curve for the UI
    val_pred_path = Path(out_cfg.get("val_predictions_path", "reports/val_predictions.csv"))
    curve_path    = Path(out_cfg.get("threshold_curve_path", "reports/threshold_curve.csv"))
    val_df = pd.DataFrame({"y_true": y_test.to_numpy(), "proba": proba})
    val_pred_path.parent.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(val_pred_path, index=False)
    curve.to_csv(curve_path, index=False)


    # ---- Save artifacts ----
    model_path = Path(out_cfg.get("model_path", "models/pipeline.joblib"))
    metrics_path = Path(out_cfg.get("metrics_path", "reports/metrics.json"))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model   → {model_path}")
    print(f"Saved metrics → {metrics_path}")
    print(f"ROC-AUC: {roc_auc:.3f} | PR-AUC: {pr_auc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()
    main(args.config)

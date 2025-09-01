# ui/app.py
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------
# Paths / Artifacts
# ---------------------------
MODEL_PATH   = Path("models/pipeline.joblib")
FEAT_PATH    = Path("models/feature_names.json")
BG_PATH      = Path("models/background.joblib")
META_PATH    = Path("models/meta.json")
METRICS_JSON = Path("reports/metrics.json")
VAL_PRED_CSV = Path("reports/val_predictions.csv")
CURVE_CSV    = Path("reports/threshold_curve.csv")

st.set_page_config(page_title="Churn Radar", layout="wide")
st.title("Churn Radar — MVP + Explainability")

def cost_optimal_threshold(curve_df: pd.DataFrame, cost_fp: float, cost_fn: float) -> tuple[float, float]:
    c = curve_df.copy()
    c["expected_cost"] = c["fp"] * cost_fp + c["fn"] * cost_fn
    best = c.sort_values(["expected_cost", "threshold"]).iloc[0]
    return float(best["threshold"]), float(best["expected_cost"])

# ---------------------------
# Load model & companions
# ---------------------------
if not MODEL_PATH.exists():
    st.error("Model not found. Run `make train` first.")
    st.stop()

pipe = joblib.load(MODEL_PATH)
feature_names = json.load(open(FEAT_PATH)) if FEAT_PATH.exists() else None
X_bg = joblib.load(BG_PATH) if BG_PATH.exists() else None
meta = json.load(open(META_PATH)) if META_PATH.exists() else {"model": "logreg"}
model_name = (meta.get("model") or "logreg").lower()

# ---------------------------
# Sidebar: Threshold control (safe session_state usage)
# ---------------------------
st.sidebar.header("Decision Threshold")

# (1) compute recommendations if available
rec_thr = {"best_f1": None, "precision>=0.60": None, "precision>=0.70": None}
if METRICS_JSON.exists():
    try:
        recs = json.load(open(METRICS_JSON))
        rec_thr.update(recs.get("thresholds", {}))
    except Exception:
        pass

# (2) ensure default BEFORE any widgets are created
if "thr" not in st.session_state:
    st.session_state["thr"] = 0.50

# (3) callbacks to update state (these run BEFORE the slider is instantiated on the rerun)
def _set_thr(val: float | None):
    if val is not None:
        st.session_state["thr"] = float(val)

# Quick-pick buttons FIRST (so their on_click can set state, then slider is created below)
qp_cols = st.sidebar.columns(3)
qp_cols[0].button(
    "Best F1",
    disabled=(rec_thr["best_f1"] is None),
    on_click=_set_thr,
    args=(rec_thr["best_f1"],),
)
qp_cols[1].button(
    "Prec ≥ 0.60",
    disabled=(rec_thr["precision>=0.60"] is None),
    on_click=_set_thr,
    args=(rec_thr["precision>=0.60"],),
)
qp_cols[2].button(
    "Prec ≥ 0.70",
    disabled=(rec_thr["precision>=0.70"] is None),
    on_click=_set_thr,
    args=(rec_thr["precision>=0.70"],),
)

# Reset button (also BEFORE the slider)
st.sidebar.button("Reset 0.50", on_click=_set_thr, args=(0.50,))

# (4) single slider bound to 'thr' (no 'value=' arg)
st.sidebar.slider("Threshold for 'Churn = 1'", 0.0, 1.0, key="thr")

# --- Cost / ROI tuner ---
with st.sidebar.expander("Cost / ROI tuner", expanded=False):
    cost_fp = st.number_input("Cost of contacting a non-churner (FP)", min_value=0.0, value=1.0, step=0.1)
    cost_fn = st.number_input("Cost of losing a churner (FN)", min_value=0.0, value=10.0, step=1.0)

    if CURVE_CSV.exists():
        try:
            curve_df = pd.read_csv(CURVE_CSV)
            thr_opt, exp_cost = cost_optimal_threshold(curve_df, cost_fp, cost_fn)
            st.caption(f"Recommended threshold: **{thr_opt:.2f}** (expected validation cost ≈ **{exp_cost:.2f}**)")

            if st.button("Apply cost-optimal threshold"):
                st.session_state["thr"] = thr_opt
        except Exception as e:
            st.warning(f"Could not compute cost-optimal threshold: {e}")
    else:
        st.info("Train first to generate the threshold curve (reports/threshold_curve.csv).")

threshold = float(st.session_state["thr"])

# ---------------------------
# SHAP helper
# ---------------------------
def compute_shap_for_row(df_row: pd.DataFrame):
    try:
        import shap
    except Exception as e:
        st.warning(f"Install SHAP to see explanations. ({e})")
        return None, None

    # Transform the row using the fitted preprocessor
    X_row = pipe.named_steps["preprocess"].transform(df_row)
    clf = pipe.named_steps["clf"]

    # Pick explainer
    try:
        if model_name == "xgb":
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X_row)
            if isinstance(shap_vals, list):
                sv = shap_vals[1]  # class 1
            else:
                sv = shap_vals
        else:
            # Generic fallback (KernelExplainer) requires a background sample
            if X_bg is None:
                return None, None
            explainer = shap.KernelExplainer(clf.predict_proba, X_bg)
            shap_vals = explainer.shap_values(X_row)
            if isinstance(shap_vals, list):
                sv = shap_vals[1]
            else:
                sv = shap_vals
    except Exception as e:
        st.info(f"SHAP explanation unavailable: {e}")
        return None, None

    sv = np.array(sv).reshape(-1)
    fn = feature_names if feature_names is not None else [f"f{i}" for i in range(len(sv))]
    return sv, fn

# ---------------------------
# Single prediction
# ---------------------------
st.subheader("Single Prediction")
with st.form("single_pred"):
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("gender", ["Female", "Male"])
        senior = st.selectbox("SeniorCitizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
    with c2:
        tenure = st.number_input("tenure (months)", min_value=0, max_value=100, value=12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        monthly = st.number_input("MonthlyCharges", min_value=0.0, max_value=300.0, value=70.0)
    with c3:
        total = st.number_input("TotalCharges", min_value=0.0, max_value=10000.0, value=800.0)
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        phone = st.selectbox("PhoneService", ["Yes", "No"])
    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "Contract": contract,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])
    proba = float(pipe.predict_proba(row)[:, 1][0])
    pred = int(proba >= threshold)
    st.metric("Churn Probability", f"{proba:.2%}")
    st.write(f"Decision (threshold={threshold:.2f}): **{'Churn' if pred==1 else 'Stay'}**")

    with st.expander("Explain this prediction (SHAP)", expanded=False):
        sv, fn = compute_shap_for_row(row)
        if sv is not None:
            import matplotlib.pyplot as plt
            idx = np.argsort(np.abs(sv))[::-1][:10]
            names = [fn[i] for i in idx]
            vals = sv[idx]
            fig = plt.figure(figsize=(6, 4))
            y_pos = np.arange(len(names))[::-1]
            plt.barh(y_pos, vals[y_pos])
            plt.yticks(y_pos, names)
            plt.title("Top SHAP Feature Impacts (customer)")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No SHAP explanation available.")

st.markdown("---")

# ---------------------------
# Batch prediction
# ---------------------------
st.subheader("Batch Upload (CSV)")
file = st.file_uploader("Upload a CSV with feature columns", type=["csv"])
if file is not None:
    df = pd.read_csv(file)
    try:
        probs = pipe.predict_proba(df)[:, 1]
        df_out = df.copy()
        df_out["churn_proba"] = probs
        df_out["pred"] = (df_out["churn_proba"] >= threshold).astype(int)
        st.success("Predictions added as `churn_proba` and `pred`.")
        st.dataframe(df_out.head(20), use_container_width=True)
        st.download_button("Download predictions CSV", df_out.to_csv(index=False), file_name="predictions.csv")
    except Exception as e:
        st.error(f"Prediction failed. Check your columns. Error: {e}")
        st.write("Tip: start by uploading a few rows from the Telco CSV with the target column dropped.")

st.markdown("---")

# ---------------------------
# Validation metrics @ current threshold
# ---------------------------
with st.expander("Validation metrics at current threshold", expanded=False):
    if VAL_PRED_CSV.exists():
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
        vdf = pd.read_csv(VAL_PRED_CSV)
        y_true = vdf["y_true"].to_numpy()
        proba = vdf["proba"].to_numpy()
        thr = float(st.session_state.thr)
        y_pred = (proba >= thr).astype(int)

        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        m1, m2, m3 = st.columns(3)
        m1.metric("Precision", f"{p:.3f}")
        m2.metric("Recall", f"{r:.3f}")
        m3.metric("F1", f"{f1:.3f}")

        st.write(f"Confusion matrix at threshold = {thr:.2f}")
        cm_df = pd.DataFrame([[tn, fp], [fn, tp]],
                             columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"])
        st.dataframe(cm_df)

        # Optional: show precision/recall curves vs threshold
        if CURVE_CSV.exists():
            import matplotlib.pyplot as plt
            curve = pd.read_csv(CURVE_CSV)
            fig1 = plt.figure(figsize=(6, 3))
            plt.plot(curve["threshold"], curve["precision"], label="Precision")
            plt.plot(curve["threshold"], curve["recall"], label="Recall")
            plt.axvline(thr, linestyle="--")
            plt.xlabel("Threshold"); plt.ylabel("Score"); plt.title("Precision/Recall vs Threshold")
            plt.legend(); plt.tight_layout()
            st.pyplot(fig1)
    else:
        st.info("Run training first to populate validation predictions and threshold curve.")

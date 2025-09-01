# src/data.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
import kagglehub
from typing import Tuple

TELCO_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
EXCLUDE = {"customerID"}  # drop obvious IDs

def fetch_telco_dataset() -> Path:
    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    csv_path = Path(path) / TELCO_FILE
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected dataset not found at {csv_path}")
    return csv_path

def load_and_clean(
    csv_path: str | None = None,
    target: str = "Churn",
    positive_label: str = "Yes",
) -> Tuple[pd.DataFrame, pd.Series]:
    if not csv_path or not Path(csv_path).exists():
        csv_path = fetch_telco_dataset()

    df = pd.read_csv(csv_path)

    # Known numeric cleanup
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Normalize target, keep only rows with target present
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataset.")
    df[target] = df[target].astype(str).str.strip()
    df = df.dropna(subset=[target])

    # Drop ID columns
    df = df.drop(columns=[c for c in df.columns if c in EXCLUDE], errors="ignore")

    # Build X, y (leave feature NaNs for imputers in preprocess)
    y_raw = df[target]
    X = df.drop(columns=[target])

    # Yes/No -> 1/0 (case-insensitive)
    y = (y_raw.str.lower() == str(positive_label).strip().lower()).astype(int)

    # Sanity check BEFORE split
    vc = y.value_counts(dropna=False)
    print("Telco y class counts (before split):", vc.to_dict())
    if y.nunique() < 2:
        print("Unique raw Churn values:", sorted(y_raw.dropna().unique().tolist())[:10])
        raise ValueError("Only one class found in target after mapping.")

    return X, y

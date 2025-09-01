# src/preprocess.py
from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build a ColumnTransformer that:
      - imputes + scales numeric features
      - imputes + one-hot encodes categorical features
    Returns: (preprocessor, numeric_columns, categorical_columns)
    """
    # Treat object/category/bool as categorical
    categorical: List[str] = list(X.select_dtypes(include=["object", "category", "bool"]).columns)
    numeric: List[str] = list(X.select_dtypes(include=["number"]).columns)

    # Numeric pipeline: median impute + standardize
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline: most_frequent impute + one-hot encode (dense output for sklearn 1.5+)
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric),
            ("cat", cat_pipe, categorical),
        ]
    )

    return preprocessor, numeric, categorical

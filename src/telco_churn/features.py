from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ID_COL = "customerID"
TARGET_COL = "churn_flag"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# We'll treat SeniorCitizen as categorical (0/1) for interpretability
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]


def build_preprocessor() -> ColumnTransformer:
    """
    Returns a preprocessing transformer for mixed numeric/categorical data.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def make_xy(df: pd.DataFrame):
    """
    Split a cleaned telco dataframe into X, y, and ids.
    """
    ids = df[ID_COL]
    y = df[TARGET_COL]
    X = df.drop(columns=[ID_COL, "Churn", TARGET_COL])
    return X, y, ids

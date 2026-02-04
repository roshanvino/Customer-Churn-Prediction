from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class TelcoSchema:
    id_col: str = "customerID"
    target_col: str = "Churn"
    target_flag_col: str = "churn_flag"
    total_charges_col: str = "TotalCharges"


SCHEMA = TelcoSchema()


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the raw Telco churn CSV from disk.
    Keeps raw data unchanged (cleaning happens in clean_telco()).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    return pd.read_csv(csv_path)


def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardise the Telco churn dataset for modeling.
    Minimal, purposeful cleaning only:
      - create churn_flag (Yes/No -> 1/0)
      - convert TotalCharges to numeric (handle blank strings)
      - basic trimming of whitespace for object cols
      - validate customerID uniqueness
    """
    df = df.copy()

    # --- Validate required columns
    required = {SCHEMA.id_col, SCHEMA.target_col, SCHEMA.total_charges_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # --- Strip whitespace from all object columns (safe + helps with blanks)
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()

    # --- customerID uniqueness check
    if df[SCHEMA.id_col].duplicated().any():
        dup_count = int(df[SCHEMA.id_col].duplicated().sum())
        raise ValueError(f"Duplicate {SCHEMA.id_col} values found: {dup_count}")

    # --- Target: Yes/No -> 1/0
    churn_map = {"Yes": 1, "No": 0}
    df[SCHEMA.target_flag_col] = df[SCHEMA.target_col].map(churn_map)

    if df[SCHEMA.target_flag_col].isna().any():
        bad_vals = sorted(df.loc[df[SCHEMA.target_flag_col].isna(), SCHEMA.target_col].unique())
        raise ValueError(f"Unexpected values in {SCHEMA.target_col}: {bad_vals}")

    # --- TotalCharges: blank strings -> NaN -> numeric
    # Kaggle Telco dataset often has blank TotalCharges for tenure=0 customers.
    df[SCHEMA.total_charges_col] = pd.to_numeric(df[SCHEMA.total_charges_col], errors="coerce")

    # --- SeniorCitizen: treat as categorical-like 0/1 int (already int in your df)
    # Leave as-is; it will be handled later via feature lists/pipeline.

    return df

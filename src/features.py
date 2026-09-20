"""
Feature engineering for the house-price prediction project.

This module defines helpers that:
1. Compute TotalBathrooms from individual bathroom columns.
2. Select the three final predictors: GrLivArea, BedroomAbvGr, TotalBathrooms.
3. Validate that required columns exist and contain valid values.

Feature definitions (verified against data_description.txt):
- GrLivArea: Above-ground living area in square feet.
- BedroomAbvGr: Number of bedrooms above ground (excludes basement bedrooms).
- TotalBathrooms: FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath.
  Half baths (powder rooms) count as 0.5 each. Basement bathrooms are included
  so the feature captures total bathroom capacity, not just above-ground.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# Exact column names expected in the Kaggle dataset
TARGET_COL = "SalePrice"
ID_COL = "Id"

# The three bathroom components that sum into TotalBathrooms
BATH_COMPONENTS: List[str] = [
    "FullBath",
    "HalfBath",
    "BsmtFullBath",
    "BsmtHalfBath",
]

# Final predictor columns (order matters for the pipeline)
FEATURE_COLS: List[str] = ["GrLivArea", "BedroomAbvGr", "TotalBathrooms"]


def compute_total_bathrooms(df: pd.DataFrame) -> pd.Series:
    """Compute TotalBathrooms = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath.

    If *any* component is missing the resulting total is NaN (not zero).
    This ensures we impute missing bathroom counts from training data rather
    than silently assuming zero bathrooms.
    """
    full = df["FullBath"].astype(float)
    half = df["HalfBath"].astype(float) * 0.5
    bsmt_full = df["BsmtFullBath"].astype(float)
    bsmt_half = df["BsmtHalfBath"].astype(float) * 0.5
    total = full + half + bsmt_full + bsmt_half
    # Propagate NaN: if any component was NaN the sum is NaN
    any_missing = df[BATH_COMPONENTS].isnull().any(axis=1)
    total = total.where(~any_missing, other=np.nan)
    return total


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with the three final predictors ready for the pipeline.

    Accepts two equivalent input shapes:
    1. Raw dataset columns: GrLivArea, BedroomAbvGr plus the four bathroom
       components (FullBath, HalfBath, BsmtFullBath, BsmtHalfBath) -> computes
       TotalBathrooms (missing component => NaN, imputed downstream).
    2. Pre-built features: GrLivArea, BedroomAbvGr and an existing
       TotalBathrooms column (used by the Streamlit app, where the user enters
       the bathroom total directly).
    """
    required_core = ["GrLivArea", "BedroomAbvGr"]
    missing_core = [c for c in required_core if c not in df.columns]
    if missing_core:
        raise ValueError(
            f"Missing required columns in input DataFrame: {missing_core}. "
            "Check that you are using the correct CSV file."
        )

    has_components = all(c in df.columns for c in BATH_COMPONENTS)
    has_total = "TotalBathrooms" in df.columns
    if not has_components and not has_total:
        raise ValueError(
            "Missing bathroom columns in input DataFrame. Provide either the four "
            f"bathroom components {BATH_COMPONENTS} or a pre-computed 'TotalBathrooms' column."
        )

    out = df[["GrLivArea", "BedroomAbvGr"]].copy()
    if has_components:
        out["TotalBathrooms"] = compute_total_bathrooms(df)
    else:
        out["TotalBathrooms"] = df["TotalBathrooms"].astype(float)
    return out


def validate_dataframe(df: pd.DataFrame, role: str = "data") -> None:
    """Raise a clear error if the DataFrame is missing critical columns or has no rows."""
    if len(df) == 0:
        raise ValueError(f"The {role} DataFrame is empty (zero rows).")
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in the {role} DataFrame. "
            f"Available columns include: {list(df.columns[:10])}..."
        )


def dataset_hash(csv_path: Path) -> str:
    """Return a short SHA-256 hash of a CSV file for reproducibility tracking."""
    h = hashlib.sha256()
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def training_ranges(df: pd.DataFrame) -> dict:
    """Return min/max of each feature from the training data for out-of-range warnings."""
    features = build_features(df)
    return {
        "GrLivArea": {"min": float(features["GrLivArea"].min()), "max": float(features["GrLivArea"].max())},
        "BedroomAbvGr": {"min": float(features["BedroomAbvGr"].min()), "max": float(features["BedroomAbvGr"].max())},
        "TotalBathrooms": {"min": float(features["TotalBathrooms"].min()), "max": float(features["TotalBathrooms"].max())},
    }


def validate_predictor_inputs(df: pd.DataFrame, allow_missing: bool = False) -> None:
    """Validate the three predictor columns before running inference.

    Rules (applied to every non-missing value):
    - GrLivArea: finite and strictly positive (square footage).
    - BedroomAbvGr: finite, non-negative integer.
    - TotalBathrooms: finite, non-negative, and a multiple of 0.5.

    ``allow_missing``: when True, NaN cells are permitted (the preprocessing
    imputer fills them during predict). The Streamlit app passes
    ``allow_missing=False`` so that user inputs are fully validated up front.
    Raises ValueError with row indices describing every problem found.
    """
    if len(df) == 0:
        raise ValueError("Cannot validate predictions for an empty input.")

    input_df = df.copy()
    problems = []

    for col in ("GrLivArea", "BedroomAbvGr", "TotalBathrooms"):
        if col not in input_df.columns:
            raise ValueError(f"Missing predictor column '{col}' before validation.")

    for idx, row in input_df.iterrows():
        values = {c: row[c] for c in ("GrLivArea", "BedroomAbvGr", "TotalBathrooms")}
        for col, val in values.items():
            if pd.isna(val):
                if not allow_missing:
                    problems.append(f"row {idx}: {col} is missing")
                continue
            if not np.isfinite(val):
                problems.append(f"row {idx}: {col} is not finite")

        area = values["GrLivArea"]
        if not pd.isna(area) and np.isfinite(area) and area <= 0:
            problems.append(f"row {idx}: GrLivArea must be positive, got {area}")

        beds = values["BedroomAbvGr"]
        if not pd.isna(beds) and np.isfinite(beds):
            if beds != int(beds):
                problems.append(f"row {idx}: BedroomAbvGr must be an integer, got {beds}")
            if beds < 0:
                problems.append(f"row {idx}: BedroomAbvGr must be non-negative, got {beds}")

        baths = values["TotalBathrooms"]
        if not pd.isna(baths) and np.isfinite(baths):
            if baths < 0:
                problems.append(f"row {idx}: TotalBathrooms must be non-negative, got {baths}")
            if abs(baths * 2 - round(baths * 2)) > 1e-6:
                problems.append(f"row {idx}: TotalBathrooms must be in 0.5 increments, got {baths}")

    if problems:
        raise ValueError("Invalid predictor values:\n  " + "\n  ".join(problems[:20]))

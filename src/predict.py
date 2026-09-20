"""
Generate Kaggle submission predictions from data/test.csv.

Usage:
    python -m src.predict

This script:
1. Loads the trained pipeline from models/pipeline.joblib
2. Reads data/test.csv
3. Applies the same feature engineering used during training
4. Saves outputs/submission.csv with columns Id and SalePrice
5. Reports row count, missing predictions, and non-finite values
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

OUTPUTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
from src.features import (
    ID_COL,
    TARGET_COL,
    build_features,
    validate_predictor_inputs,
)


def run() -> None:
    """Main prediction entry point."""
    print("=" * 60)
    print("  House Price Prediction — Generate Submission")
    print("=" * 60)

    # 1. Load the trusted pipeline -------------------------------------------
    pipeline_path = MODELS_DIR / "pipeline.joblib"
    if not pipeline_path.exists():
        print(
            f"ERROR: Model artifact not found at {pipeline_path}\n"
            "Run training first:  python -m src.train"
        )
        sys.exit(1)

    pipeline = joblib.load(pipeline_path)
    print(f"Loaded pipeline from {pipeline_path}")

    # 2. Load test data ------------------------------------------------------
    test_path = DATA_DIR / "test.csv"
    if not test_path.exists():
        print(
            f"ERROR: Test data not found at {test_path}\n"
            "Download from: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data\n"
            "Place test.csv inside the data/ directory."
        )
        sys.exit(1)

    test_df = pd.read_csv(test_path)
    print(f"Loaded {len(test_df)} rows from test.csv")

    # Preserve original order
    test_df = test_df.sort_values(ID_COL).reset_index(drop=True)

    # 3. Validate and predict -------------------------------------------------
    # Shared validation (missing predictors are allowed; the pipeline imputer fills them)
    features_df = build_features(test_df)
    validate_predictor_inputs(features_df, allow_missing=True)
    predictions = pipeline.predict(test_df)

    # 4. Build submission DataFrame ------------------------------------------
    submission = pd.DataFrame({
        ID_COL: test_df[ID_COL],
        TARGET_COL: predictions,
    })

    # 5. Quality checks ------------------------------------------------------
    n_rows = len(submission)
    n_missing = submission[TARGET_COL].isnull().sum()
    n_nonfinite = int((~np.isfinite(submission[TARGET_COL])).sum())
    n_nonpositive = int((submission[TARGET_COL] <= 0).sum())

    print(f"\n--- Submission Quality Report ---")
    print(f"  Total rows          : {n_rows}")
    print(f"  Missing predictions : {n_missing}")
    print(f"  Non-finite values   : {n_nonfinite}")
    print(f"  Non-positive values : {n_nonpositive}")

    if n_nonpositive > 0:
        print(
            f"\nWARNING: {n_nonpositive} predictions are non-positive. "
            "These are reported as-is (not clipped) to highlight model limitations. "
            "Linear regression can produce unrealistic estimates for unusual inputs."
        )

    if n_missing > 0:
        print(f"\nWARNING: {n_missing} rows have missing predictions.")

    if n_nonfinite > 0:
        print(f"\nWARNING: {n_nonfinite} rows have non-finite (NaN/Inf) predictions.")

    # 6. Save ----------------------------------------------------------------
    out_path = OUTPUTS_DIR / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSaved submission to {out_path}")
    print("Note: Kaggle test.csv has no SalePrice labels — local metrics cannot be computed.")


if __name__ == "__main__":
    run()

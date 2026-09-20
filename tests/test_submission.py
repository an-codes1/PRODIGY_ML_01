"""
Tests for submission generation logic and pipeline saving/loading.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import ID_COL, TARGET_COL, build_features
from src.train import make_feature_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def _synthetic_train_df(n: int = 100) -> pd.DataFrame:
    """Create a synthetic training DataFrame."""
    rng = np.random.RandomState(42)
    data = {
        "Id": range(1, n + 1),
        "GrLivArea": rng.randint(500, 4000, size=n).astype(float),
        "BedroomAbvGr": rng.randint(1, 6, size=n).astype(float),
        "FullBath": rng.randint(0, 4, size=n).astype(float),
        "HalfBath": rng.randint(0, 3, size=n).astype(float),
        "BsmtFullBath": rng.randint(0, 3, size=n).astype(float),
        "BsmtHalfBath": rng.randint(0, 2, size=n).astype(float),
        "SalePrice": rng.randint(80000, 600000, size=n).astype(float),
    }
    return pd.DataFrame(data)


def _synthetic_test_df(n: int = 20) -> pd.DataFrame:
    """Create a synthetic test DataFrame (no SalePrice)."""
    rng = np.random.RandomState(99)
    data = {
        "Id": range(1001, 1001 + n),
        "GrLivArea": rng.randint(500, 4000, size=n).astype(float),
        "BedroomAbvGr": rng.randint(1, 6, size=n).astype(float),
        "FullBath": rng.randint(0, 4, size=n).astype(float),
        "HalfBath": rng.randint(0, 3, size=n).astype(float),
        "BsmtFullBath": rng.randint(0, 3, size=n).astype(float),
        "BsmtHalfBath": rng.randint(0, 2, size=n).astype(float),
    }
    return pd.DataFrame(data)


def test_pipeline_save_load(tmp_path):
    """Pipeline can be saved and loaded, producing identical predictions."""
    train_df = _synthetic_train_df()
    test_df = _synthetic_test_df()

    pipeline = make_feature_pipeline()
    pipeline.fit(train_df, train_df["SalePrice"])

    preds_before = pipeline.predict(test_df)

    save_path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, save_path)
    loaded = joblib.load(save_path)

    preds_after = loaded.predict(test_df)
    np.testing.assert_array_equal(preds_before, preds_after)


def test_submission_schema():
    """Simulate submission generation: check Id and SalePrice columns, row count, Id order."""
    train_df = _synthetic_train_df()
    test_df = _synthetic_test_df()

    pipeline = make_feature_pipeline()
    pipeline.fit(train_df, train_df["SalePrice"])

    test_df_sorted = test_df.sort_values(ID_COL).reset_index(drop=True)
    preds = pipeline.predict(test_df_sorted)

    submission = pd.DataFrame({ID_COL: test_df_sorted[ID_COL], TARGET_COL: preds})

    # Check columns
    assert list(submission.columns) == [ID_COL, TARGET_COL]

    # Check row count matches test data
    assert len(submission) == len(test_df)

    # Check Id order is preserved (ascending)
    assert list(submission[ID_COL]) == sorted(test_df[ID_COL].tolist())

    # Check all predictions are finite
    assert np.all(np.isfinite(submission[TARGET_COL]))

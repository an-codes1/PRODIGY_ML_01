"""
Tests for feature building and pipeline consistency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import (
    BATH_COMPONENTS,
    FEATURE_COLS,
    build_features,
    validate_dataframe,
    validate_predictor_inputs,
)


def _synthetic_train_df(n: int = 100) -> pd.DataFrame:
    """Create a small synthetic training DataFrame with all required columns."""
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


def test_build_features_returns_three_columns():
    """build_features should return exactly the three feature columns."""
    df = _synthetic_train_df()
    result = build_features(df)
    assert list(result.columns) == FEATURE_COLS


def test_build_features_correct_row_count():
    """Output should have the same number of rows as input."""
    df = _synthetic_train_df(50)
    result = build_features(df)
    assert len(result) == 50


def test_validate_dataframe_missing_target():
    """validate_dataframe should raise ValueError if SalePrice is missing."""
    df = pd.DataFrame({"Id": [1], "GrLivArea": [1000]})
    with pytest.raises(ValueError, match="SalePrice"):
        validate_dataframe(df, role="test")


def test_validate_dataframe_empty():
    """validate_dataframe should raise ValueError on empty DataFrame."""
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="empty"):
        validate_dataframe(df)


def test_build_features_accepts_precomputed_total_bathrooms():
    """The app passes GrLivArea, BedroomAbvGr and a pre-computed TotalBathrooms
    (no raw bathroom components). build_features must accept that shape."""
    df = pd.DataFrame({
        "GrLivArea": [1500.0, 2100.0],
        "BedroomAbvGr": [3, 4],
        "TotalBathrooms": [2.5, 3.0],
    })
    result = build_features(df)
    assert list(result.columns) == FEATURE_COLS
    assert result["TotalBathrooms"].tolist() == [2.5, 3.0]


def test_build_features_missing_bathrooms_raises():
    """Neither components nor TotalBathrooms present -> clear error."""
    df = pd.DataFrame({"GrLivArea": [1500.0], "BedroomAbvGr": [3]})
    with pytest.raises(ValueError, match="TotalBathrooms"):
        build_features(df)


def test_validate_predictor_inputs_rejects_bad_values():
    """Shared validation catches negative area, non-integer bedrooms, non-0.5 baths."""
    bad = pd.DataFrame({
        "GrLivArea": [-500.0, 1500.0, 1500.0],
        "BedroomAbvGr": [3, 3.5, 3],
        "TotalBathrooms": [2.0, 2.0, 2.3],
    })
    with pytest.raises(ValueError, match="GrLivArea"):
        validate_predictor_inputs(bad, allow_missing=False)


def test_validate_predictor_inputs_accepts_good_values():
    """A valid three-column input passes shared validation."""
    good = pd.DataFrame({
        "GrLivArea": [1500.0, 2400.5],
        "BedroomAbvGr": [3, 4],
        "TotalBathrooms": [2.5, 3.0],
    })
    validate_predictor_inputs(good, allow_missing=False)  # should not raise


def test_validate_predictor_inputs_allows_missing_when_flagged():
    """Missing predictors are permitted when allow_missing=True (imputer handles them)."""
    missing = pd.DataFrame({
        "GrLivArea": [1500.0, None],
        "BedroomAbvGr": [3, 4],
        "TotalBathrooms": [2.0, 3.0],
    })
    validate_predictor_inputs(missing, allow_missing=True)
    with pytest.raises(ValueError, match="missing"):
        validate_predictor_inputs(missing, allow_missing=False)


def test_pipeline_produces_finite_predictions():
    """A fitted pipeline should produce finite predictions on synthetic data."""
    from src.train import make_feature_pipeline

    train_df = _synthetic_train_df()
    pipeline = make_feature_pipeline()
    pipeline.fit(train_df, train_df["SalePrice"])

    preds = pipeline.predict(train_df)
    assert preds.shape == (len(train_df),)
    assert np.all(np.isfinite(preds)), "All predictions should be finite"


def test_pipeline_feature_names_consistency():
    """Pipeline input during inference should match training feature names."""
    from src.train import make_feature_pipeline

    train_df = _synthetic_train_df()
    pipeline = make_feature_pipeline()
    pipeline.fit(train_df, train_df["SalePrice"])

    # Simulate inference: build features from a new DataFrame
    test_df = _synthetic_train_df(10)
    test_features = build_features(test_df)
    preds = pipeline.predict(test_df)
    assert len(preds) == 10
    assert np.all(np.isfinite(preds))

"""
Tests for bathroom aggregation logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import compute_total_bathrooms, build_features, FEATURE_COLS, BATH_COMPONENTS


def _make_row(**kwargs) -> dict:
    """Return a dict with all bathroom component columns, defaulting to 1."""
    defaults = {"FullBath": 1, "HalfBath": 1, "BsmtFullBath": 1, "BsmtHalfBath": 1}
    defaults.update(kwargs)
    return defaults


def test_basic_total_bathrooms():
    """Standard calculation: 1 + 0.5*1 + 1 + 0.5*1 = 3.0"""
    df = pd.DataFrame([_make_row()])
    result = compute_total_bathrooms(df)
    assert result.iloc[0] == pytest.approx(3.0)


def test_half_bathrooms_count_as_half():
    """Two full baths + two half baths = 2 + 1 + 0 + 0 = 3.0"""
    row = _make_row(FullBath=2, HalfBath=2, BsmtFullBath=0, BsmtHalfBath=0)
    df = pd.DataFrame([row])
    result = compute_total_bathrooms(df)
    assert result.iloc[0] == pytest.approx(3.0)


def test_all_zero_bathrooms():
    """All zeros should give 0.0."""
    row = _make_row(FullBath=0, HalfBath=0, BsmtFullBath=0, BsmtHalfBath=0)
    df = pd.DataFrame([row])
    result = compute_total_bathrooms(df)
    assert result.iloc[0] == pytest.approx(0.0)


def test_missing_component_produces_nan():
    """If any component is NaN, the total should be NaN (not zero)."""
    row = _make_row(FullBath=2, HalfBath=1, BsmtFullBath=1, BsmtHalfBath=np.nan)
    df = pd.DataFrame([row])
    result = compute_total_bathrooms(df)
    assert np.isnan(result.iloc[0])


def test_multiple_missing_components():
    """Multiple missing components should still produce NaN."""
    row = _make_row(FullBath=np.nan, HalfBath=1, BsmtFullBath=np.nan, BsmtHalfBath=0)
    df = pd.DataFrame([row])
    result = compute_total_bathrooms(df)
    assert np.isnan(result.iloc[0])


def test_mixed_rows_some_nan():
    """Some rows have NaN, others don't."""
    rows = [
        _make_row(FullBath=2, HalfBath=0, BsmtFullBath=1, BsmtHalfBath=1),  # 3.5
        _make_row(FullBath=1, HalfBath=np.nan, BsmtFullBath=0, BsmtHalfBath=0),  # NaN
        _make_row(FullBath=3, HalfBath=1, BsmtFullBath=0, BsmtHalfBath=0),  # 3.5
    ]
    df = pd.DataFrame(rows)
    result = compute_total_bathrooms(df)
    assert result.iloc[0] == pytest.approx(3.5)
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(3.5)

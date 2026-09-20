"""
Train a multiple-linear-regression model for house-price prediction.

Usage:
    python -m src.train

This script:
1. Loads data/train.csv
2. Builds features (GrLivArea, BedroomAbvGr, TotalBathrooms)
3. Splits 80/20 (random_state=42)
4. Trains a Pipeline with MedianImputer + LinearRegression
5. Evaluates on validation split (MAE, RMSE, R2)
6. Compares against a DummyRegressor baseline
7. Saves metrics, plots, and the final pipeline fitted on ALL training rows
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MaxNLocator
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# Path helpers – everything is relative to the project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Ensure output directories exist
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Import our feature helpers
sys.path.insert(0, str(PROJECT_ROOT))
from src.features import (
    FEATURE_COLS,
    TARGET_COL,
    build_features,
    dataset_hash,
    training_ranges,
    validate_dataframe,
)


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load a CSV and basic validation."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}\n"
            "Download from: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data\n"
            "Place train.csv inside the data/ directory."
        )
    df = pd.read_csv(csv_path)
    return df


def make_feature_pipeline() -> Pipeline:
    """Build a scikit-learn Pipeline:
       1. FunctionTransformer to select/build the three features.
       2. SimpleImputer(strategy='median') to fill missing values.
       3. LinearRegression model.
    """
    pipeline = Pipeline([
        ("features", FunctionTransformer(build_features, validate=False)),
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LinearRegression()),
    ])
    return pipeline


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, and R-squared."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}


def _dollar_ticks(value: float, _pos) -> str:
    """Format axis ticks like $100k, $200k; keep negative signs on residuals."""
    if abs(value) / 1000 >= 1:
        sign = "-" if value < 0 else ""
        return f"{sign}${abs(value) / 1000:,.0f}k"
    return f"${value:,.0f}"


def save_validation_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    outputs_dir: Path,
) -> None:
    """Generate and save actual-vs-predicted and residual plots.

    Plot axes use compact dollar tick labels ($100k, $200k, ...) so titles
    and labels stay readable at desktop and mobile widths.
    """
    residuals = y_true - y_pred
    dollar = FuncFormatter(_dollar_ticks)

    # --- Actual vs Predicted ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, y_true, alpha=0.5, edgecolors="k", linewidths=0.3)
    lo = min(y_true.min(), y_pred.min()) * 0.9
    hi = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Predicted Sale Price ($)")
    ax.set_ylabel("Actual Sale Price ($)")
    ax.set_title("Actual vs Predicted Sale Price")
    ax.xaxis.set_major_formatter(dollar)
    ax.yaxis.set_major_formatter(dollar)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.legend()
    fig.tight_layout()
    fig.savefig(outputs_dir / "actual_vs_predicted.png", dpi=120)
    plt.close(fig)

    # --- Residual plot ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, residuals, alpha=0.5, edgecolors="k", linewidths=0.3)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted Sale Price ($)")
    ax.set_ylabel("Residual ($)")
    ax.set_title("Residual Plot")
    ax.xaxis.set_major_formatter(dollar)
    ax.yaxis.set_major_formatter(dollar)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    fig.tight_layout()
    fig.savefig(outputs_dir / "residual_plot.png", dpi=120)
    plt.close(fig)


def run() -> None:
    """Main training entry point."""
    print("=" * 60)
    print("  House Price Prediction — Training")
    print("=" * 60)

    # 1. Load data ----------------------------------------------------------
    csv_path = DATA_DIR / "train.csv"
    df = load_data(csv_path)
    validate_dataframe(df, role="training")
    print(f"Loaded {len(df)} rows from {csv_path.name}")

    dhash = dataset_hash(csv_path)
    print(f"Dataset hash: {dhash}")

    # 2. Train / validation split (80/20) -----------------------------------
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Split: {len(train_df)} training / {len(val_df)} validation rows")

    # 3. Build and fit pipeline on training split ----------------------------
    pipeline = make_feature_pipeline()
    pipeline.fit(train_df, train_df[TARGET_COL])
    print("Pipeline fitted on training split.")

    # 4. Evaluate on validation split ----------------------------------------
    val_pred = pipeline.predict(val_df)
    metrics = evaluate(val_df[TARGET_COL].values, val_pred)
    print(f"\n--- Validation Metrics ---")
    print(f"  MAE  : ${metrics['MAE']:,.2f}")
    print(f"  RMSE : ${metrics['RMSE']:,.2f}")
    print(f"  R²   : {metrics['R2']:.4f}")

    # 4b. Dummy baseline (predicts training mean) ----------------------------
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(train_df[TARGET_COL].values.reshape(-1, 1), train_df[TARGET_COL].values)
    dummy_pred = dummy.predict(val_df[TARGET_COL].values.reshape(-1, 1))
    dummy_metrics = evaluate(val_df[TARGET_COL].values, dummy_pred)
    print(f"\n--- Dummy Baseline (predicts mean) ---")
    print(f"  MAE  : ${dummy_metrics['MAE']:,.2f}")
    print(f"  RMSE : ${dummy_metrics['RMSE']:,.2f}")
    print(f"  R²   : {dummy_metrics['R2']:.4f}")

    # 4c. Coefficients -------------------------------------------------------
    final_model = pipeline.named_steps["model"]
    print(f"\n--- Model Coefficients ---")
    print(f"  Intercept: ${final_model.intercept_:,.2f}")
    for name, coef in zip(FEATURE_COLS, final_model.coef_):
        print(f"  {name:20s}: ${coef:,.2f}")
    print("  (Coefficients describe associations, not causal effects.)")

    # 5. Save validation artifacts -------------------------------------------
    metrics_all = {
        "linear_regression": metrics,
        "dummy_baseline": dummy_metrics,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "random_state": 42,
    }
    with open(OUTPUTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_all, f, indent=2)
    print(f"\nSaved metrics to {OUTPUTS_DIR / 'metrics.json'}")

    save_validation_plots(val_df[TARGET_COL].values, val_pred, OUTPUTS_DIR)
    print(f"Saved plots to {OUTPUTS_DIR}")

    # 6. Refit on ALL labeled rows for the production model ------------------
    print("\nRefitting pipeline on all labeled training rows...")
    final_pipeline = make_feature_pipeline()
    final_pipeline.fit(df, df[TARGET_COL])

    # 7. Save pipeline and metadata -----------------------------------------
    ranges_ = training_ranges(df)
    metadata = {
        "feature_columns": FEATURE_COLS,
        "target": TARGET_COL,
        "training_rows": len(df),
        "dataset_hash": dhash,
        "training_ranges": ranges_,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "dependencies": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": __import__("sklearn").__version__,
        },
        "coefficients": {
            "intercept": float(final_model.intercept_),
            **{name: float(coef) for name, coef in zip(FEATURE_COLS, final_model.coef_)},
        },
    }

    joblib.dump(final_pipeline, MODELS_DIR / "pipeline.joblib")
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved final pipeline to {MODELS_DIR / 'pipeline.joblib'}")
    print(f"Saved metadata to {MODELS_DIR / 'metadata.json'}")
    print("\nTraining complete.")


if __name__ == "__main__":
    run()

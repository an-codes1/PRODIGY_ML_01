"""
Streamlit demo for the House Price Predictor.

Run with:
    python -m streamlit run app.py

This app:
- Explains the project and feature definitions.
- Provides inputs for GrLivArea, BedroomAbvGr, and TotalBathrooms.
- Loads the trained pipeline and predicts SalePrice.
- Displays validation metrics and plots from training.
- Includes help text, input validation, and educational notes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

sys.path.insert(0, str(PROJECT_ROOT))
from src.features import (
    FEATURE_COLS,
    validate_predictor_inputs,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("House Price Predictor")
st.caption("An educational estimate based on historical Ames, Iowa housing data.")

# ---------------------------------------------------------------------------
# Project explanation
# ---------------------------------------------------------------------------
st.markdown("""
### About this project
This app predicts house sale prices using **multiple linear regression** with three features
from the Ames Housing dataset. It is designed as an educational tool for learning ML basics.

**Features used:**
| Feature | Description |
|---|---|
| **GrLivArea** | Above-ground living area in square feet |
| **BedroomAbvGr** | Number of bedrooms above ground (does not count basement bedrooms) |
| **TotalBathrooms** | FullBath + 0.5 × HalfBath + BsmtFullBath + 0.5 × BsmtHalfBath |

> **Note:** This model only uses three features. It omits major price influences such as
> location, neighborhood, condition, age, and construction quality. Predictions are
> estimates, not appraisals.
""")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
# The model artifact is a project-generated file at a fixed application path.
# joblib deserialization can execute arbitrary code, so only trusted files from
# this repository are loaded — never user-uploaded or remote joblib/pickle files.
@st.cache_resource(show_spinner=False)
def load_trusted_pipeline(path: Path):
    return joblib.load(path)


pipeline_path = MODELS_DIR / "pipeline.joblib"
metadata_path = MODELS_DIR / "metadata.json"

pipeline = None
metadata = None

if pipeline_path.exists():
    pipeline = load_trusted_pipeline(pipeline_path)
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
else:
    st.warning(
        "Model not found. Train the model locally first:\n\n"
        "```\npython -m src.train\n```\n\n"
        "Then redeploy from the repository."
    )

# ---------------------------------------------------------------------------
# Training ranges (for out-of-range warnings)
# ---------------------------------------------------------------------------
training_ranges = {}
if metadata and "training_ranges" in metadata:
    training_ranges = metadata["training_ranges"]

# ---------------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------------
st.markdown("### Enter house features")

col1, col2, col3 = st.columns(3)

with col1:
    gr_liv_area = st.number_input(
        "Living area (sq ft)",
        min_value=100.0,
        max_value=20000.0,
        value=1500.0,
        step=50.0,
        help="Above-ground living area in square feet (GrLivArea). Typical range: 400–5000 sq ft.",
    )

with col2:
    bedrooms = st.number_input(
        "Bedrooms (above ground)",
        min_value=0,
        max_value=15,
        value=3,
        step=1,
        help="Number of bedrooms above ground (BedroomAbvGr). Does not include basement bedrooms.",
    )

with col3:
    bathrooms = st.number_input(
        "Total bathrooms",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help=(
            "Computed as: FullBath + 0.5 × HalfBath + BsmtFullBath + 0.5 × BsmtHalfBath. "
            "Enter the total yourself (e.g., 2.5 = 2 full + 1 half bath)."
        ),
    )

# ---------------------------------------------------------------------------
# Input validation & range check
# ---------------------------------------------------------------------------
out_of_range = []
if training_ranges:
    r = training_ranges.get("GrLivArea", {})
    if gr_liv_area < r.get("min", 0) or gr_liv_area > r.get("max", 99999):
        out_of_range.append(f"Living area ({gr_liv_area} sq ft) is outside training range [{r.get('min')}, {r.get('max')}]")
    r = training_ranges.get("BedroomAbvGr", {})
    if bedrooms < r.get("min", 0) or bedrooms > r.get("max", 99):
        out_of_range.append(f"Bedroom count ({bedrooms}) is outside training range [{r.get('min')}, {r.get('max')}]")
    r = training_ranges.get("TotalBathrooms", {})
    if bathrooms < r.get("min", 0) or bathrooms > r.get("max", 99):
        out_of_range.append(f"Bathroom count ({bathrooms}) is outside training range [{r.get('min')}, {r.get('max')}]")

if out_of_range:
    for msg in out_of_range:
        st.warning(msg)

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if st.button("Predict Price", type="primary"):
    if pipeline is None:
        st.error("Model not loaded. Train locally first with `python -m src.train`.")
    else:
        input_df = pd.DataFrame([{
            "GrLivArea": float(gr_liv_area),
            "BedroomAbvGr": int(bedrooms),
            "TotalBathrooms": float(bathrooms),
        }])

        try:
            # Validate user inputs in the shared prediction logic too,
            # not only through widget bounds.
            validate_predictor_inputs(input_df, allow_missing=False)
            prediction = pipeline.predict(input_df)[0]
        except ValueError as exc:
            st.error(f"Invalid inputs: {exc}")
        except Exception:
            # Keep internals hidden from public visitors; details go to server logs.
            print("Prediction failed unexpectedly.", exc_info=True)
            sys.stdout.flush()
            st.error("Prediction failed. Please review your inputs and try again.")
        else:
            if not np.isfinite(prediction):
                st.error("The model returned a non-finite estimate. Please review your inputs.")
            elif prediction <= 0:
                st.error(
                    f"The model produced a non-positive estimate: **${prediction:,.0f}**. "
                    "Linear regression can produce unrealistic values when inputs are unusual. "
                    "This highlights the limitations of a simple three-feature model."
                )
            else:
                st.metric("Predicted Sale Price", f"${prediction:,.0f}")

        st.caption(
            "This is an educational estimate based on historical Ames, Iowa housing data. "
            "It does not estimate current Indian property prices."
        )

# ---------------------------------------------------------------------------
# Validation metrics & plots
# ---------------------------------------------------------------------------
st.markdown("---")
metrics_path = OUTPUTS_DIR / "metrics.json"
label = "### Model Evaluation"
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)
    n_val = metrics.get("n_val")
    label = f"### Validation results: {n_val} houses (80/20 split)"
st.markdown(label)

if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)

    lr = metrics.get("linear_regression", {})
    dm = metrics.get("dummy_baseline", {})

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Linear Regression**")
        st.write(f"MAE:  ${lr.get('MAE', 'N/A'):,.2f}")
        st.write(f"RMSE: ${lr.get('RMSE', 'N/A'):,.2f}")
        st.write(f"R²:   {lr.get('R2', 'N/A')}")
    with col_b:
        st.markdown("**Dummy Baseline (predicts mean)**")
        st.write(f"MAE:  ${dm.get('MAE', 'N/A'):,.2f}")
        st.write(f"RMSE: ${dm.get('RMSE', 'N/A'):,.2f}")
        st.write(f"R²:   {dm.get('R2', 'N/A')}")

    # Display plots
    avp_path = OUTPUTS_DIR / "actual_vs_predicted.png"
    res_path = OUTPUTS_DIR / "residual_plot.png"

    if avp_path.exists():
        st.image(str(avp_path), caption="Actual vs Predicted Sale Price", width="stretch")
    if res_path.exists():
        st.image(str(res_path), caption="Residual Plot", width="stretch")
else:
    st.info("No evaluation metrics found. Run `python -m src.train` to generate them.")

# ---------------------------------------------------------------------------
# Educational notes
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
### How it works

**Multiple Linear Regression** finds the best-fitting linear relationship:

```
Price = intercept
     + coefficient_1 × GrLivArea
     + coefficient_2 × BedroomAbvGr
     + coefficient_3 × TotalBathrooms
```

The model minimises the sum of squared errors between predicted and actual prices.

**Evaluation metrics:**
- **MAE (Mean Absolute Error):** Average dollar difference between predicted and actual price.
- **RMSE (Root Mean Squared Error):** Like MAE but penalises large errors more heavily.
- **R² (R-squared):** Fraction of price variance explained by the model (1.0 = perfect, 0.0 = no better than predicting the mean).

**Data leakage** occurs when information from the test set accidentally influences training.
This project avoids it by fitting preprocessing only on the training split.
""")

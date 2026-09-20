# PRODIGY_ML_01 — House Price Predictor

**Machine Learning Task 01** (Prodigy InfoTech internship): a beginner-friendly
multiple linear regression project that predicts house sale prices
using three features from the **Ames Housing dataset** (Kaggle).

---

## Project Objective

Implement multiple linear regression to predict house prices from:

1. **Square footage** — `GrLivArea` (above-ground living area in sq ft)
2. **Number of bedrooms** — `BedroomAbvGr` (bedrooms above ground; excludes basement bedrooms)
3. **Number of bathrooms** — `TotalBathrooms` (computed from four source columns)

### Feature Definitions

| Feature | Column(s) | Unit | Notes |
|---|---|---|---|
| Living Area | `GrLivArea` | sq ft | Above-ground living area only |
| Bedrooms | `BedroomAbvGr` | count | Above-ground bedrooms only (no basement bedrooms) |
| Total Bathrooms | `FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath` | count (0.5 increments) | Half baths count as 0.5; basement bathrooms are included |

> The bathroom feature includes basement bathrooms and counts half-bathrooms (powder rooms with toilet + sink but no tub/shower) as 0.5 each.

---

## Dataset

Source: [Kaggle House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

The dataset is the **Ames Housing dataset** (De Cock, 2011), made available for the
Kaggle *House Prices — Advanced Regression Techniques* competition.

**Licensing / provenance:** the Kaggle competition files came from the
*house-prices-advanced-regression-techniques* dataset; its redistribution license is
not explicitly stated on Kaggle, so this repository does **not** redistribute the raw
`train.csv` / `test.csv` (they are excluded via `.gitignore` and must be downloaded
again). The Ames Housing data is also published under the GPL-2 license in the CRAN
`AmesHousing` package for those who need a clearly-licensed copy. A nearest-neighbor
match on house characteristics is **not** applied — the repository ships only model
artifacts and outputs derived from the data, not the raw records.

### Manual Download Steps

1. Go to the Kaggle link above.
2. Click **Data** → **Download** (you may need a free Kaggle account).
3. Extract the ZIP file.
4. Place `train.csv`, `test.csv`, and `data_description.txt` inside the `data/` directory.

The `data/` directory should look like:

```
data/
  train.csv
  test.csv
  data_description.txt
```

> **Note:** The dataset files are large and are excluded from Git via `.gitignore`.

---

## Setup (Windows PowerShell)

```powershell
# 1. Clone this project
git clone https://github.com/an-codes1/PRODIGY_ML_01.git
cd PRODIGY_ML_01

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
.\venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt

# 5. Place dataset files in data/ (see Dataset section above)
```

> **Tested runtime:** Python 3.12.10 with pandas 2.3.3, numpy 1.26.4,
> scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2, joblib 1.6.0,
> streamlit 1.64.0, pytest 9.1.1. `requirements.txt` pins these versions.

---

## Running the Project

### Train the model
```powershell
python -m src.train
```

### Generate Kaggle submission
```powershell
python -m src.predict
```

### Launch the Streamlit app on Streamlit Community Cloud
This repository is deployed on [Streamlit Community Cloud](https://share.streamlit.io).
**Live demo:** https://house-price-predictor-egpecb3xtdh8nuek2tgvgr.streamlit.app/

> To deploy: sign in to share.streamlit.io with your GitHub account, click
> **New app**, select the `PRODIGY_ML_01` repository and `main` branch, set the main
> file to `app.py`, and (optional) choose Python **3.12** under **Advanced settings**
> before clicking **Deploy**. The app loads the committed `models/pipeline.joblib` and
> `outputs/*.png` — no dataset download is needed on the server.

### Run tests
```powershell
python -m pytest
```

### Launch Streamlit demo (local only)
```powershell
python -m streamlit run app.py
```
This starts a **local** demo that opens in your browser at `http://localhost:8501`.
It is not deployed online; it runs only on your machine until you close the terminal.

### Optional: run the Jupyter notebook
The notebook `notebooks/house_price_walkthrough.ipynb` reuses the same project
functions for a step-by-step explanation. Install Jupyter first:

```powershell
pip install jupyter ipykernel
jupyter notebook notebooks\house_price_walkthrough.ipynb
```
(`pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn` are already in
`requirements.txt`.)

---

## Project Structure

```
PRODIGY_ML_01/
├── app.py                         # Streamlit local demo
├── requirements.txt               # Python dependencies (pinned)
├── README.md                      # This file
├── .gitignore                     # Ignores venv, raw data, caches
│
├── .github/
│   └── workflows/
│       └── ci.yml                 # Tests + dependency audit on push/PR
│
├── src/
│   ├── __init__.py
│   ├── features.py                # Feature engineering & validation
│   ├── train.py                   # Training, evaluation, model saving
│   └── predict.py                 # Generate Kaggle submission CSV
│
├── tests/
│   ├── __init__.py
│   ├── test_bathroom.py           # Tests for bathroom aggregation
│   ├── test_pipeline.py           # Tests for feature building & pipeline
│   └── test_submission.py         # Tests for submission schema & saving
│
├── notebooks/
│   └── house_price_walkthrough.ipynb  # Educational Jupyter notebook
│
├── data/                          # Place train.csv, test.csv here
├── models/                        # Saved pipeline & metadata (generated)
└── outputs/                       # Metrics, plots, submission (generated)
```

---

## What Each File Does

| File | Purpose |
|---|---|
| `app.py` | Streamlit app for interactive prediction demo; loads only the trusted committed model artifact, validates inputs, and hides stack traces from visitors |
| `src/features.py` | Defines feature columns, builds TotalBathrooms, validates DataFrames, shared predictor-input validation for both `app.py` and `src/predict.py` |
| `src/train.py` | Loads data, splits, trains, evaluates, saves pipeline and metrics |
| `src/predict.py` | Loads trained pipeline, predicts on test.csv, saves submission |
| `.github/workflows/ci.yml` | Runs dataset-independent tests and a dependency audit (`pip-audit`) on push/PR |
| `tests/test_bathroom.py` | Tests bathroom math (half baths, missing values) |
| `tests/test_pipeline.py` | Tests feature pipeline output consistency |
| `tests/test_submission.py` | Tests model save/load and submission schema |

---

## Machine Learning Concepts

### Multiple Linear Regression

Multiple linear regression finds the **best-fitting linear relationship**:

```
Price = intercept
     + coefficient_1 × GrLivArea
     + coefficient_2 × BedroomAbvGr
     + coefficient_3 × TotalBathrooms
```

Each coefficient (c1, c2, c3) represents the change in price associated with a one-unit
increase in that feature, holding the others constant. These are **associations**, not
causal effects.

### Evaluation Metrics

- **MAE (Mean Absolute Error):** The average dollar difference between predicted and
  actual price. A MAE of $30,000 means the model's predictions are off by ~$30,000 on average.

- **RMSE (Root Mean Squared Error):** Similar to MAE but penalises large errors more
  heavily. An RMSE of $40,000 means large mistakes are common.

- **R² (R-squared):** The fraction of price variance the model explains. R² = 0.80 means
  80% of price variation is captured. R² = 0.0 means the model is no better than
  predicting the average price. **R² is not a percentage accuracy** — it's a proportion
  of explained variance.

### Data Leakage

Data leakage happens when test-set information accidentally influences training. This
project prevents leakage by:

1. Splitting data before any preprocessing.
2. Fitting the imputer only on the training split.
3. Using a scikit-learn `Pipeline` so preprocessing and prediction are bundled.

---

## Limitations

- Only **three features** out of 79 available in the dataset. Major price drivers like
  **location** (Neighborhood), **condition**, **construction quality**, **lot size**,
  **age**, and **garage** are omitted.
- The model is trained on **Ames, Iowa** housing data from 2006–2010. It does **not**
  estimate current Indian property prices.
- Linear regression assumes a straight-line relationship, which may not hold for all
  price ranges.
- No log transformation of the target, so the model may struggle with the long tail of
  expensive homes.

---

## Actual Measured Results

Measured on the validation split (292 rows, 20% of the 1460-row training set, `random_state=42`):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Linear Regression (3 features)** | $34,395.80 | $51,222.58 | 0.6579 |
| **Dummy Baseline (predicts mean)** | $62,575.93 | $87,619.03 | −0.0009 |

The linear model clearly beats the dummy baseline, confirming the three features carry
real signal. An R² of 0.66 means the model explains about 66% of price variance.
Note that R² is not "66% accurate" — it is a proportion of explained variance.

Model coefficients (final model refit on all 1,460 rows):

| Term | Value |
|---|---|
| Intercept | $36,976 |
| GrLivArea | $94.66 / sq ft |
| BedroomAbvGr | −$23,244 / bedroom |
| TotalBathrooms | $30,178 / bathroom |

> **Why is the bedroom coefficient negative?** This does **not** mean bedrooms make
> houses less valuable. Coefficients are measured *conditional on the other predictors*:
> for two houses with the same living area and the same bathrooms, the fitted line says a
> model with one extra above-ground bedroom is associated with a lower predicted price.
> Bedrooms are strongly correlated with living area, so once area is held fixed the
> model assigns the extra variation to area. This is a statistical association within
> this sample, not a causal claim about house markets.

Metrics and plots are regenerated by `python -m src.train` and stored in `outputs/`.

---

## Demonstration Walkthrough

1. **Train:** Run `python -m src.train`. The script splits data, fits a pipeline,
   evaluates on validation data, and saves `models/pipeline.joblib`.

2. **Evaluate:** Check `outputs/metrics.json` for MAE, RMSE, and R². The Streamlit app
   displays these with plots.

3. **Predict:** Run `python -m src.predict` to generate `outputs/submission.csv`.

4. **Demo:** Run `python -m streamlit run app.py` to interact with the model in a browser.

---

## Viva Questions

### Q1: What features does this model use and why those three?
**A:** GrLivArea (living area), BedroomAbvGr (bedrooms above ground), and TotalBathrooms
(computed from FullBath, HalfBath, BsmtFullBath, BsmtHalfBath). These are intuitive
features that have strong linear relationships with price. The bathroom feature includes
basement bathrooms and counts half-baths as 0.5.

### Q2: Why use a Pipeline instead of separate preprocessing steps?
**A:** A Pipeline bundles preprocessing and the model into one object. This prevents data
leakage (fitting the imputer on test data) and makes saving/loading/deploying a single
step.

### Q3: What is data leakage and how is it avoided here?
**A:** Data leakage is when test-set information leaks into training. Here, we split
data first, fit preprocessing (imputation) only on the training split, and evaluate on
an untouched validation set.

### Q4: Why compare against a DummyRegressor?
**A:** The DummyRegressor predicts the training mean for every house (R² = 0). Our model
must beat this baseline to prove it has learned something useful. If MAE or R² is only
slightly better, the model has limited practical value.

### Q5: Why might the model produce non-positive price predictions?
**A:** Linear regression fits an unbounded straight line. For unusual inputs (very small
area, zero bedrooms), the equation can produce negative values. This highlights the
limitation of linear models — real prices are always positive.

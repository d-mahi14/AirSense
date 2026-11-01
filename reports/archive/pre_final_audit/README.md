# AirSense

**Air Quality Prediction and Analysis Using Machine Learning**

A local Streamlit dashboard for the saved AirSense classifiers and the actual
experiments completed in prompts 11 and 12. Prompt 13 adds the application only:
**no retraining, retuning, replacement scores or new forecasting claims.**

## Windows PowerShell: install and launch

Install **64-bit Python 3.12** from python.org with the Python launcher, then
extract `AirSense.zip`. Open PowerShell in the extracted **AirSense** folder
(the folder containing `app.py` and `requirements.txt`):

```powershell
# Replace this path with your extracted project folder.
Set-Location "C:\path\to\AirSense"

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open **http://localhost:8501**. Stop with **Ctrl+C**. Virtual-environment activation
is not required, so these commands do not require changing PowerShell's execution
policy. For subsequent launches:

```powershell
Set-Location "C:\path\to\AirSense"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The saved pipelines were produced with Python 3.12, scikit-learn 1.8.0 and
XGBoost 3.4.1. Dependencies are pinned to the verified environment. Loading a
pipeline with a different scikit-learn version is rejected with installation
instructions rather than silently risking incompatible inference.

## Five application sections

| Section | Included behavior |
| --- | --- |
| Home | Introduction, problem, objectives, workflow and metrics read from verified experimental results |
| Air Quality Prediction | Baseline/tuned model selection, fitted feature schema, date-derived calendar inputs, numeric validation, estimated category and supported probabilities |
| Model Comparison | Four-model table, accuracy/macro-F1 charts, all confusion matrices, per-class precision/recall/F1 and a CSV download; baseline and tuned views |
| Pollution Analysis | Actual pollutant distributions, descriptive statistics, pairwise Spearman/Pearson correlations, saved RF/XGBoost importance and saved XGBoost SHAP |
| Research and Methodology | Dataset/source, primary survey, algorithm foundations, preprocessing, chronological evaluation, real results, limitations and references |

## Prediction contract

- The saved model exposes its fitted feature order. The UI asks only for that
  model's supported pollutant predictors. Current artifacts use PM2.5, PM10,
  NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene and Xylene.
- Year, Month, Day, DayOfWeek, IsWeekend and Season are derived automatically
  from the measurement date using `extract_temporal_features`, the training
  implementation. Numeric pollutant columns are float64; temporal columns keep
  the training extractor's dtypes. The complete fitted pipeline handles all
  clipping, imputation, encoding and scaling.
- Enter **daily measurements in the original source-dataset units**. No unit
  conversions are performed. The source CSV does not carry a complete machine-
  readable unit dictionary, so the dashboard does not invent one. Hourly readings
  and mismatched sensor units are not interchangeable with the daily training data.
- Readings must be finite and nonnegative. Application upper bounds are 120% of
  the observed training maximum per feature, rounded upward, and are displayed
  alongside the training median/maximum. They are engineering input limits,
  not statutory AQI thresholds or sensor-validity guarantees. Values exceeding
  the observed training maximum generate an extrapolation warning.
- A blank reading remains missing and is imputed by the original saved pipeline.
  The imputed fields are disclosed. All-blank samples are rejected. Explicit NaN,
  infinity, negative, nonnumeric, out-of-range and extra weather fields are rejected
  by the prediction service. Widget limits also prevent invalid numeric entry.
- Measurement dates must lie between 2015 and today. Dates after the training
  period receive a distribution-shift warning; entering a later date does not
  turn classification into forecasting.
- **Load a training example** fills the form with one real, fully observed
  historical training row. Its city and date are identified. It is not live data.
- DT, RF and XGBoost expose class probabilities. They are aligned using the
  artifact's `classes_`, not an assumed output order. These probabilities are
  uncalibrated, not validated confidence or health-risk estimates. This SVM was
  trained without probability estimation and reports that limitation explicitly.

**Predictions are ML estimates, not official government AQI readings.** The app
estimates a six-class, same-day category; it does not forecast future pollution
or compute an official continuous AQI measurement. No city or weather inputs
are synthesized or added.

## Saved artifacts and data

Keep the supplied folder structure:

```text
AirSense/
  app.py
  README.md
  requirements.txt
  src/prediction_service.py
  src/ui_components.py
  models/*.joblib
  models/selection_manifest.json
  reports/evaluation_results.json
  reports/feature_analysis/*.csv
  reports/feature_analysis/sampling_manifest.json
  data/raw/archive (3)/city_day.csv
```

All eight supplied pipelines (four baselines/four tuned) are available for
prediction. Only the selected pipeline is loaded, and `st.cache_resource`
reuses it across refreshes with path/mtime/size/checksum keys. Models are never
fitted by the app. Data/report reads use `st.cache_data`. Saved SHAP is displayed
rather than recalculated, and histograms are aggregated before plotting.

`reports/evaluation_results.json` is the performance source of truth. The app
checks its frozen-selection hash, class order, confusion-matrix dimensions,
support and score ranges. A matching model checksum is checked before loading.
Missing/stale/corrupt artifacts produce visible errors instead of substitute
models or fabricated metrics. Feature plots must match a verified baseline
model's hash. Missing SHAP is explicitly marked unavailable.

The raw daily CSV is used for training-derived input bounds, historical examples
and descriptive charts; the app does not load the large hourly datasets. If it
is missing, prediction inputs/descriptive charts are unavailable; saved model
comparison results still work. Data can also be located with the existing
`AIRSENSE_DATA_DIR` environment variable, pointing to a folder containing
`city_day.csv`.

## Experimental interpretation

Training ends 30 June 2019; validation covers July–December 2019; test covers
2020. Baseline Random Forest was selected on validation macro F1 in the saved
experiment. Every model's displayed performance comes from actual predictions
on the same chronological test observations. The application does not change
that experiment or select models by test scores.

The original project already inspected test outcomes. This corrected run is a
**retrospective benchmark**, not pristine prospective validation. Other limits
include one dataset/seed, pandemic/seasonal drift, rare-class support, missingness,
unmeasured weather, untested new-city transfer and uncalibrated probabilities.
Feature importance is association rather than causation; correlated predictors
can substitute for each other and attribution methods have different biases.
No direct numerical comparison with published scores is claimed.

## Primary research paper

Méndez, M., Merayo, M. G., & Núñez, M. (2023). *Machine learning algorithms to
forecast air quality: a survey*. Artificial Intelligence Review, 56,
10031–10066. https://doi.org/10.1007/s10462-023-10424-4

Sections 3.3 and 3.4 contextualize predictor groups and evaluation metrics.
AirSense's same-day classification target and macro-F1 selection are its own
experimental choices, not a claimed reproduction of the survey's studies.

Dataset: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

## Verification

The included verification record is `reports/prompt13_verification.md`; detailed
app tests are in `tests/test_prompt13_prediction.py` and
`tests/test_prompt13_dashboard.py`. It records the local server, page/form
checks, saved-artifact integrity and actual test outcomes. PowerShell syntax is
provided for Windows; execution was verified in the local Linux/Python runtime.
To run the complete test suite in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Prompts 11 and 12 remain available through their documented module commands in
`EXPERIMENT_README.md`. Running Streamlit does not run those experiments.

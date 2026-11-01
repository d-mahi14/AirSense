# AirSense

**Air Quality Prediction and Analysis Using Machine Learning**

Final submission package: a same-day, six-category AQI classifier comparison,
controlled tuning/interpretability experiment, Streamlit dashboard and research-style
paper. Predictions are estimates, not official government measurements or future
forecasts. The benchmark is retrospective because earlier work inspected test outcomes.

## Windows setup: C:\dev\AirSense

Use 64-bit **Python 3.12**. Extract the project so `app.py` is at
`C:\dev\AirSense\app.py`, then open PowerShell:

```powershell
Set-Location 'C:\dev\AirSense'
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip check
```

`requirements.txt` pins runtime, model-loading, report and audit dependencies;
`requirements-dev.txt` adds pytest. If activation is blocked by your local execution
policy, activation is optional: use `.\.venv\Scripts\python.exe` in place of `python`
in every command below. No execution-policy change is required.

## Dataset paths and selected target

The requested dataset location is `data/raw/archive(3)/`. The supplied archive
actually contains **`data/raw/archive (3)/`** (a space before the parenthesis).
Both spellings are supported; no original CSV has been moved, edited or duplicated.
Select whichever exists, without renaming the raw directory:

```powershell
$env:AIRSENSE_DATA_DIR = Join-Path $PWD 'data\raw\archive(3)'
if (-not (Test-Path (Join-Path $env:AIRSENSE_DATA_DIR 'city_day.csv'))) {
    $env:AIRSENSE_DATA_DIR = Join-Path $PWD 'data\raw\archive (3)'
}
if (-not (Test-Path (Join-Path $env:AIRSENSE_DATA_DIR 'city_day.csv'))) {
    throw 'city_day.csv was not found. Point AIRSENSE_DATA_DIR to the supplied CSV folder.'
}
```

All five originals are required by the complete dataset loader: `city_day.csv`,
`city_hour.csv`, `station_day.csv`, `station_hour.csv`, `stations.csv`. Their SHA-256
values and sizes are recorded in `reports/raw_data_reference.json` and checked
in `reports/raw_data_integrity.csv` against the original user-uploaded ZIP.
This proves preservation since upload, not independent verification of the dataset
before upload.

The modeled file is **city_day.csv**; target **AQI_Bucket** (Good, Satisfactory,
Moderate, Poor, Very Poor, Severe). Of 29,531 city-day records, 24,850 are labeled.
The source period is 1 January 2015 through 1 July 2020. Predictors are 12 pollutant
columns plus Year, Month, Day, DayOfWeek, IsWeekend and Season. AQI and target-derived
columns are excluded. No city identity or fabricated weather input is used.

## Run tests and audit the supplied artifacts

```powershell
python -m pytest -q
python -m scripts.final_audit --deep-features
```

The second command checks actual saved-model predictions, validation selection,
metrics, partitions, raw hashes, built-in importance, saved SHAP summaries and
recomputed validation permutation importance. It writes audit evidence and the
current results CSV without retraining or overwriting supplied models.

To additionally verify that all four baselines can train on the complete training
partition **in temporary storage**:

```powershell
python -m scripts.final_audit --verify-training --deep-features
```

This keeps the submitted model files and experiment results unchanged. A differing
Python patch/platform is recorded rather than treated as proof of failure; the pinned
ML libraries, protocol, data, hashes and actual predictions must still match.
A reproducibility mismatch raises an error instead of inventing success.

Exact Linux commands and actual outcomes from the final audit are in
`reports/final_audit_commands.md` and `reports/final_audit_logs/`. Windows path logic
and both directory spellings are tested; Windows installation and execution still
require your manual check on the target computer.

## Launch the application

```powershell
python -m streamlit run app.py
```

Open **http://localhost:8501**. Stop with **Ctrl+C**. No model fitting, tuning or SHAP
computation occurs on refresh. Only the selected pipeline is lazily loaded and cached.
The exact fitted preprocessing handles the ordered inputs; unsupported probabilities,
invalid entries and missing artifacts are reported explicitly.

The five sections are Home, Air Quality Prediction, Model Comparison, Pollution
Analysis, and Research and Methodology. Baseline/tuned views use actual saved results.
The form requires the selected model's pollutant features, derives calendar fields
with the training extractor, and provides a real historical training example.
Readings must use the source dataset's units and daily aggregation. Blanks are
imputed by the saved pipeline; all-blank, nonfinite, negative and out-of-range inputs
are rejected. Limits are derived from training maxima, not regulatory thresholds.
Class probabilities, where supported, are uncalibrated; this SVM has none.

## Optional: rebuild models and reports

You do not need to train anything to demonstrate the supplied app. These commands
**replace** trained artifacts and experiment reports. Preserve the current evidence first:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$backup = Join-Path $PWD (Join-Path 'backups' $stamp)
New-Item -ItemType Directory -Path $backup -Force | Out-Null
Copy-Item '.\models' -Destination (Join-Path $backup 'models') -Recurse
Copy-Item '.\reports' -Destination (Join-Path $backup 'reports') -Recurse

python -m scripts.train_all_models
python -m scripts.tune_models
python -m scripts.evaluate_all_models
python -m scripts.analyze_features
python -m scripts.generate_research_paper
python -m scripts.final_audit --deep-features
python -m pytest -q
```

`python -m scripts.run_all_models` is now a guarded compatibility entry point for
those first four stages, in that order. It no longer launches the old random-split
experiment or runs training merely when imported. Its original source is preserved
in `reports/archive/pre_final_audit/`. Paper/audit generation is explicit and separate.

Training uses dates through 30 June 2019; validation uses July-December 2019; test
uses 2020. Tuning uses three expanding-date folds with a seven-day gap and fold-local
preprocessing. All finalists are refitted on the full training split. Selection is
frozen by validation macro F1 across baselines and tuned finalists; test scores do
not change that choice. The bounded SVM search uses 6,000 training rows before its
full-data refit. This is not a complete temporal/spatial generalization study.

## Authoritative final files

| File | Purpose |
| --- | --- |
| reports/final_project_report.md | Final audit, structure, results, commands and manual tasks |
| reports/final_results.csv | All eight actual baseline/tuned results, with validation-selected candidate |
| reports/final_checklist.md | Evidence and status for all 18 requested checks |
| reports/limitations_and_future_work.md | Scientific, data, reproducibility and deployment limits |
| reports/final_audit.json | Latest machine-readable artifact audit |
| reports/evaluation_results.json | Original frozen experiment results and timing source |
| reports/test_predictions.csv | Actual predictions on the same test rows for every candidate |
| reports/research_paper.pdf | Verified 17-page research-style draft with eight figures and 13 tables |
| reports/research_paper.md | Main paper source |
| reports/references.bib | Verified bibliography |
| docs/IMPLEMENTATION_LOG.md, docs/AI_PROMPT_LOG.md, docs/AI_FAILURE_LOG.md | Appended audit and correction history |

Historical reports and the original compatibility results are preserved and clearly
marked as superseded. Use the files above for submission rather than old random-split
scores. The models/all_results.json compatibility export now reflects the actual
current experiment; its original is archived.

## Scientific limits and submission review

Baseline Random Forest is the validation-selected candidate. The paper documents
accuracy/macro-F1 disagreement, class imbalance, only 38 Severe test examples,
919 test rows from six cities absent in training, numeric calendar clipping,
source AQI extremes, missingness and unverified units, absent weather, uncalibrated
probabilities and historical test exposure. A new independent dataset is needed
for confirmation; causal, official AQI, future forecasting, statistical superiority
and broad new-city claims are unsupported.

Primary reference: Méndez, Merayo and Núñez (2023), *Machine learning algorithms to
forecast air quality: a survey*, Artificial Intelligence Review 56, 10031-10066.
DOI: https://doi.org/10.1007/s10462-023-10424-4. The publisher and full paper were verified.

Before submission, check Windows installation/activation and all five pages in your
browser, confirm authorship/affiliation and any college template, and review the
research claims and historical material. The existing paper is a research-style
draft, not a publication-readiness guarantee. No Git remote, branch, history, push,
publish or deployment action is part of this audit.


## Windows follow-up correction (1 November 2026)

User Windows run: 161 passed, 3 failed. Matching pinned versions ruled out a
simple dependency mismatch. Added native XGBoost JSON bundles and a shared
checksum-verified loader, without retraining or altering model scores. See
reports/windows_portability_fix.md (or windows_portability_fix.md within reports).
Earlier 164-test counts are historical Linux results; Windows confirmation of
this correction remains pending.

# AirSense final project report

**Prompt 15 | Audit completed 1 November 2026 | Awaiting user review**

The corrected project was inspected from raw files through fitted preprocessing,
trained artifacts, actual predictions, feature outputs, dashboard, tests and research
report. This is a retrospective same-day AQI classification project, not a future
forecasting system or official government AQI service. Software checks do not remove
its scientific limitations.

## 1. Final folder structure

| Location | Important contents |
| --- | --- |
| Root | app.py; README.md; requirements.txt; requirements-dev.txt; requirements-paper.txt; requirements-experiment-lock.txt |
| data/raw/archive (3)/ | Original city_day.csv, city_hour.csv, station_day.csv, station_hour.csv, stations.csv |
| src/ | data_loader.py; preprocessing.py; feature_engineering.py; pipeline.py; data_split.py; experiment.py; evaluation.py; tuning.py; explainability.py; prediction_service.py; ui_components.py; models/ |
| scripts/ | train_all_models.py; tune_models.py; evaluate_all_models.py; analyze_features.py; generate_research_paper.py; final_audit.py; guarded run_all_models.py |
| models/ | Eight unchanged fitted .joblib pipelines; configurations; training/tuning/selection manifests; current compatibility results |
| tests/ | Loading, preprocessing, model, experiment, prediction, dashboard, paper and final submission tests |
| reports/ | Final audit/report/checklist/results; raw hashes; original experiment outputs; Markdown sources and existing research_paper.pdf |
| reports/figures/ | EDA, model comparison, feature analysis and research-paper figures |
| reports/final_audit_logs/ | Exact command ledger, pytest output, dependency check, artifact audit, isolated training and server evidence |
| reports/archive/pre_final_audit/ | Preserved versions of replaced legacy entry point/results/report/README and modified supporting files |
| docs/ | Appended implementation, prompt and failure logs |

The requested Windows project path is `C:\dev\AirSense\`. The actual supplied
raw folder includes a space; both that spelling and `archive(3)` resolve through
the loader. No dataset folder was renamed or copied merely to fit the requested spelling.

## 2. Important files created or updated

Required final deliverables: `final_project_report.md`, `final_results.csv`,
`limitations_and_future_work.md`, `final_checklist.md`, `README.md`, tests and
all three `docs/` logs. Supporting evidence: `final_audit.json`,
`final_audit_commands.md`, `raw_data_reference.json`, `raw_data_integrity.csv`,
`final_audit_logs/`, and `scripts/final_audit.py`.

A complete PDF already exists: `reports/research_paper.pdf`, 17 pages, eight figures,
13 tables and the requested research sections. It was reopened and its SHA-256
rechecked; no new PDF was necessary and the verified research paper was not changed.
Its author/affiliation fields remain intentionally unassigned pending user confirmation.

Corrections made in this final audit:

- Preserved the old random-split run_all_models.py and replaced its active entry point
  with the corrected train/tune/frozen-evaluate/analyze sequence under a main guard.
  Importing it no longer trains or mutates AIRSENSE_DATA_DIR.
- Preserved the earlier all_results.json and exported actual current baseline values
  into the compatibility file; marked the old model_results.md as historical.
  The source evaluation_results.json, test predictions, model binaries and original
  timing measurements were not altered to improve apparent performance.
- Added real City-Date overlap detection to audit_data_split; previously its
  documentation claimed an overlap check that the implementation did not perform.
- Added document/audit dependencies to the pinned runtime list and documented all
  setup/activation, path, test, backup, training and application commands.

## 3. Actual model comparison results

Train 16,645 city-days through 30 June 2019; validation 3,784 in July-December 2019;
test 4,421 in 2020. Selection remains **baseline Random Forest**, chosen using
validation macro F1 = 0.774868. Every result below is copied from the verified
actual experiment, with all eight models' current predictions matched to the
saved rows. Values are displayed to four decimals; final_results.csv retains
recorded precision, timings, search cost and status.

| model | variant | selected_on_validation | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | baseline | True | 0.8077 | 0.7942 | 0.7743 | 0.7826 | 0.8070 |
| XGBoost | baseline | False | 0.8095 | 0.8218 | 0.7348 | 0.7690 | 0.8057 |
| SVM | baseline | False | 0.7473 | 0.7226 | 0.7232 | 0.7212 | 0.7484 |
| Decision Tree | baseline | False | 0.7351 | 0.6980 | 0.7583 | 0.7162 | 0.7394 |
| Random Forest | tuned | False | 0.7980 | 0.7727 | 0.7767 | 0.7724 | 0.7986 |
| XGBoost | tuned | False | 0.8100 | 0.8183 | 0.7301 | 0.7640 | 0.8054 |
| SVM | tuned | False | 0.7505 | 0.7240 | 0.7320 | 0.7261 | 0.7519 |
| Decision Tree | tuned | False | 0.7349 | 0.6940 | 0.7635 | 0.7184 | 0.7394 |

Baseline XGBoost has slightly higher test accuracy than baseline RF, but lower
macro F1. RF remains the validation-selected candidate. No test result was used
to change that selection during this audit. Tuning gains and negative deltas are
retained; there is no new score optimization for the submission.

## 4. Actual test and reproducibility outcomes

- Full latest test command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q`.
- **Actual outcome: 164 passed in 20.32s, exit 0.** No skipped/failed cases
  were represented as passes. The eight final-audit cases extend the preceding suite.
- `python -m pip check` returned `No broken requirements found.`, exit 0.
- Five original CSV byte sizes and SHA-256 values match the original upload.
- All eight actual model pipelines reproduce stored validation/test predictions,
  aggregate/per-class metrics and confusion matrices on the same test identities.
- Numeric clipping bounds, medians, scaler means and variances match independent
  reconstruction using training data only. Saved hyperparameters and search CV
  records match the fitted finalists.
- RF/XGBoost built-in importance agrees with saved tables; validation permutation
  was recomputed and matched. Saved per-sample SHAP values aggregate to the saved
  SHAP table. Full TreeSHAP and hyperparameter search were not rerun in this audit.
- Headless local Streamlit health and app endpoints both returned HTTP 200; the
  health body was `ok`. The server was stopped after checks. AppTest covers all
  five pages and prediction/error paths; this is not cross-browser visual QA.

The full-training reproduction used temporary storage and did not overwrite the
submitted artifacts:

| model | train_rows | validation_rows | reproduced_val_f1 |
| --- | --- | --- | --- |
| decision_tree | 16645 | 3784 | 0.7068 |
| random_forest | 16645 | 3784 | 0.7749 |
| svm | 16645 | 3784 | 0.6664 |
| xgboost | 16645 | 3784 | 0.7661 |

Exact commands and logs are in final_audit_commands.md and final_audit_logs/.
Training was executed through the isolated audit command; normal training CLI
commands are supplied for user reproduction and were not used to overwrite this
submission. Actual execution was Linux/Python 3.12, not Windows.

## 5. Known limitations

Historical test outcomes had already been inspected, so these are retrospective
results rather than pristine independent confirmation. The test contains 919 rows
from six cities absent in training; time and city-coverage shifts are mixed.
Severe has only 38 test examples. Missingness is substantial; units and extreme
source AQI values need independent quality validation. Calendar numerics are clipped,
including 2020 Year to the fitted 2019 upper bound. Weather is absent, probabilities
are uncalibrated, SVM probabilities are unavailable, and interpretation samples/search
budgets are small. No significance, causality, prospective forecast, geographic-transfer,
official measurement or publication-readiness claim is supported.

See limitations_and_future_work.md and the paper for details. The checklist marks
Windows execution as manual and qualifies current-code leakage/selection checks
with the historical-test caveat; it does not claim all scientific uncertainty is resolved.

## 6. Commands to run the application and experiment

From PowerShell on the target machine:

```powershell
Set-Location 'C:\dev\AirSense'
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip check

$env:AIRSENSE_DATA_DIR = Join-Path $PWD 'data\raw\archive(3)'
if (-not (Test-Path (Join-Path $env:AIRSENSE_DATA_DIR 'city_day.csv'))) {
    $env:AIRSENSE_DATA_DIR = Join-Path $PWD 'data\raw\archive (3)'
}
python -m pytest -q
python -m scripts.final_audit --deep-features
python -m streamlit run app.py
```

Open http://localhost:8501. If activation is blocked, use
`.\.venv\Scripts\python.exe` instead of `python`; an execution-policy change is
not required. These are supplied Windows instructions, not a claim they ran here.

For an optional full rebuild, first follow README's timestamped backup commands
for models/ and reports/, then run:

```powershell
python -m scripts.train_all_models
python -m scripts.tune_models
python -m scripts.evaluate_all_models
python -m scripts.analyze_features
python -m scripts.generate_research_paper
python -m scripts.final_audit --deep-features
python -m pytest -q
```

Training is unnecessary to run the supplied app. The audit can test full baseline
training safely using `python -m scripts.final_audit --verify-training --deep-features`.

## 7. Remaining manual tasks before submission

1. Execute setup, environment activation (or the direct interpreter fallback), tests,
   dataset-path checks and app launch on the intended Windows installation.
2. Review all five pages and a prediction in a real desktop/narrow-window browser.
3. Confirm paper authorship, affiliation and any college-required formatting.
4. Review methodology, retrospective-test wording, dataset attribution and final
   tables; submit the authoritative current files rather than legacy random-split scores.
5. Treat new external validation, uncertainty estimation and the listed design
   improvements as future research; do not claim they were completed.
6. Review this package before requesting changes. No push, publish, remote setup,
   branch change, Git-history rewrite or deployment was performed.

**Handoff:** final audit/documentation complete; awaiting the user's review. No
further changes will be made until requested.


## Windows follow-up correction (1 November 2026)

User Windows run: 161 passed, 3 failed. Matching pinned versions ruled out a
simple dependency mismatch. Added native XGBoost JSON bundles and a shared
checksum-verified loader, without retraining or altering model scores. See
reports/windows_portability_fix.md (or windows_portability_fix.md within reports).
Earlier 164-test counts are historical Linux results; Windows confirmation of
this correction remains pending.

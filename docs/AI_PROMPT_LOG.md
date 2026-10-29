# AirSense — AI Prompt Log

This document records the exact user requests, instructions, and prompt history across the lifecycle of the AirSense project.

---

## Stage 1: Inception, Inspection & Architectural Documentation

### Exact User Prompt Received:
```text
I am building my ML mini-project:

AirSense — Air Quality Prediction and Analysis Using Machine Learning.

PROJECT LOCATION:
C:\dev\AirSense\

DATASET LOCATION:
data/raw/archive(3)/

There are five CSV files in this directory. Inspect the actual
filenames and contents. Do not assume their schemas.

PRIMARY DATASET:
India Air Quality dataset (2015–2020), obtained from:
https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

PRIMARY RESEARCH PAPER:
Méndez, M., Merayo, M. G., and Núñez, M. (2023).
"Machine learning algorithms to forecast air quality: a survey."
Artificial Intelligence Review.
DOI: 10.1007/s10462-023-10424-4

Publisher page:
https://doi.org/10.1007/s10462-023-10424-4

MODELS:
1. Decision Tree
2. Random Forest
3. Support Vector Machine (SVM)
4. XGBoost

MAIN OBJECTIVE:
Compare these four algorithms for air-quality classification
using the actual dataset and its documented AQI categories,
if available.

ADDITIONAL OBJECTIVES:
- Perform exploratory data analysis.
- Investigate pollutant distributions and relationships.
- Compare model performance fairly.
- Explain important predictive features.
- Build a Streamlit dashboard.
- Write a research-style report based on the primary paper
  and our own experimental findings.

TECHNOLOGY:
Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib,
Seaborn, Streamlit, and SHAP if appropriate.

FIRST TASK:
Inspect the existing workspace and all five CSV files.

Create:
- PROJECT_PLAN.md
- REQUIREMENTS.md
- IMPLEMENTATION_ROADMAP.md
- reports/dataset_requirements.md

Do not train models or build the dashboard yet.

Explain the proposed architecture, available dataset files,
possible target columns, and implementation stages.

GLOBAL RULES:
- Never assume weather variables exist.
- Never invent records, metrics, citations, or findings.
- Never modify or delete the original CSV files.
- Never switch to a different primary dataset without approval.
- Never claim current-condition classification is future forecasting.
- Never push to GitHub, configure a remote, publish, or change branches.
- Do not overwrite existing user work.
- Use Windows-compatible paths and PowerShell commands.
- Keep code understandable to a student.

For each stage, maintain:
- docs/IMPLEMENTATION_LOG.md
- docs/AI_PROMPT_LOG.md
- docs/AI_FAILURE_LOG.md

Record the exact prompts used, files changed, commands executed,
test results, errors, and fixes.

Use local Git commits only if Git is already configured and
the working tree is safe. Prefix any new commit messages with
[AI:Antigravity]. Never push.

Finish by summarizing the inspection and wait for my review.
```

### Objectives Identified & Executed:
1. Conduct deep inspection of the raw directory (`data/raw/archive (3)/`) and verify the exact schemas, row counts, memory footprints, and missingness of all 5 CSV files without assumptions.
2. Confirm the presence/absence of weather variables (result: confirmed zero weather variables exist).
3. Investigate `AQI_Bucket` and `AQI` columns, matching against CPCB NAQI standards.
4. Synthesize the findings and primary reference paper (Méndez et al., 2023).
5. Produce the four required top-level architectural and requirement documents:
   - `PROJECT_PLAN.md`
   - `REQUIREMENTS.md`
   - `IMPLEMENTATION_ROADMAP.md`
   - `reports/dataset_requirements.md`
6. Maintain the three required documentation logs:
   - `docs/IMPLEMENTATION_LOG.md`
   - `docs/AI_PROMPT_LOG.md`
   - `docs/AI_FAILURE_LOG.md`
7. Halt execution before training any models or generating dashboard code, summarizing the inspection for user review.


---

## Stage 2: Detailed Dataset Audit, Inspection Script, Profile Report & Tests

### Exact User Prompt Received:
```text
Continue the AirSense project in C:\dev\AirSense.

The five original CSV files are inside:
data/raw/archive(3)/

Perform a detailed dataset audit before any model training.

Inspect every CSV and report:
- Exact filename
- Rows and columns
- Column names and data types
- Date range and timestamp format
- City and station coverage, if present
- Pollutant measurements
- AQI and AQI category fields
- Weather fields, if present
- Missing-value counts and percentages
- Duplicate records
- Invalid or implausible measurements
- Target-category distribution
- Possible overlap between files
- Whether files contain different temporal resolutions
- Whether any files are metadata rather than observations

Create:
- src/data_loader.py
- scripts/inspect_dataset.py
- reports/dataset_profile.md
- tests/test_data_loader.py

The inspection script must work on Windows and discover the
five CSV files using a configurable directory path.

Do not print or assume results without actually reading the files.

Do not merge the files blindly. Determine whether they represent
different datasets, different temporal resolutions, or related
tables.

Recommend the best file for the primary classification experiment
and explain why.

If the dataset has no weather measurements, explicitly document
that limitation. Do not fabricate weather columns.

Do not change the original files.

Run the script and relevant tests. Record actual results in
docs/IMPLEMENTATION_LOG.md and wait for my review.
```

### Objectives Identified & Executed:
1. Built `src/data_loader.py` with configurable path resolution, dedicated file loaders, constants, and documented absence of weather measurements.
2. Built `scripts/inspect_dataset.py` with Windows compatibility, argparse CLI flags (`--data-dir`, `--export-report`), chunked memory-safe ingestion, duplicate verification, and empirical statistical calculations.
3. Formatted and exported `reports/dataset_profile.md` directly via the inspection script.
4. Created `tests/test_data_loader.py` with 12 comprehensive unit tests using pytest.
5. Successfully ran the script and pytest suite (12/12 passing).
6. Updated `docs/IMPLEMENTATION_LOG.md`, `docs/AI_PROMPT_LOG.md`, and `docs/AI_FAILURE_LOG.md`.

---

## Stage 3: Theoretical Foundation & Literature Review Grounding

### Exact User Prompt Received:
```text
Build the theoretical foundation for AirSense using the
primary research paper:

Méndez, M., Merayo, M. G., and Núñez, M. (2023).
"Machine learning algorithms to forecast air quality: a survey."
Artificial Intelligence Review.
DOI: 10.1007/s10462-023-10424-4

Use the publisher page and the full paper if available.

Do not pretend to have read the full paper if its content
cannot be accessed. Clearly identify any source limitations.

Create:
- references/
- reports/literature_review.md
- reports/theoretical_background.md
- reports/research_objectives.md
- reports/references.bib

Cover:
1. Air pollution and its health/environmental relevance.
2. Air Quality Index and pollutant measurements.
3. Air-quality classification versus forecasting.
4. Machine learning approaches used in air-quality research.
5. Pollutant and meteorological predictor variables.
6. Model evaluation and generalization.
7. Interpretability and limitations.
8. How AirSense relates to the survey.

Explain these four algorithms theoretically:
- Decision Tree
- Random Forest
- Support Vector Machine
- XGBoost

Explain classification metrics:
accuracy, precision, recall, F1-score, and confusion matrix.

Explain MAE, RMSE, and R-squared only as relevant to a possible
future numerical forecasting extension.

Use original wording and proper citations.

Do not fabricate paper findings, page numbers, references,
or research gaps.

Clearly distinguish claims supported by the survey from
our planned implementation choices.

Propose a modest, defensible student-project contribution:
a controlled comparison of four models on a documented
Indian air-quality dataset, including error analysis and
feature-importance analysis.

Do not claim that this comparison alone constitutes a novel
scientific algorithm.

Run any documentation checks available and record the work.
```

### Objectives Identified & Executed:
1. Created `references/README.md` and `reports/references.bib` with verified publication records and honest source limitation statements.
2. Wrote `reports/literature_review.md` covering all 8 specified areas using original prose.
3. Wrote `reports/theoretical_background.md` with complete mathematical formulations for DT, RF, SVM, XGBoost, classification metrics, and regression metrics (for future forecasting).
4. Wrote `reports/research_objectives.md` framing a modest, defensible student contribution with testable hypotheses, explicitly disclaiming algorithmic novelty.
5. Verified all documents programmatically and re-ran test suite (`pytest` passing 12/12).
6. Maintained `docs/IMPLEMENTATION_LOG.md`, `docs/AI_PROMPT_LOG.md`, and `docs/AI_FAILURE_LOG.md`.

---

## Stage 4: Data Cleaning, Preprocessing Pipeline & Exploratory Data Analysis

### Exact User Prompt Received:
```text
Implement data cleaning and exploratory data analysis for AirSense.

First use the dataset audit to identify the selected CSV and
its actual schema.

Create:
- src/preprocessing.py
- notebooks/01_exploratory_data_analysis.ipynb
- scripts/run_eda.py
- reports/eda_report.md
- reports/figures/eda/

Analyze:
1. Missing values.
2. Duplicate records.
3. Invalid measurements.
4. Pollutant distributions.
5. AQI distribution.
6. AQI-category distribution.
7. Correlations among available numerical features.
8. City-wise comparisons if supported.
9. Temporal pollution trends if timestamps are available.

Generate suitable plots:
- Missing-value chart.
- Pollutant distributions.
- AQI-category counts.
- Correlation heatmap.
- Temporal trend chart.
- City comparisons when meaningful.

Use readable titles, labels, units, and figure captions.

Do not remove large portions of data silently.
Document every important cleaning decision.

Do not impute target labels.
Do not use the target to impute predictor features.
Do not fit preprocessing transformations on test data.

Keep original CSVs unchanged. Save derived files separately
under data/processed/ only when needed.

Run the EDA and relevant tests, and document actual findings.
```

### Objectives Identified & Executed:
1. Designed and tested `src/preprocessing.py` strictly enforcing zero target imputation, leakage-free fold fitting, and spatial-temporal median imputation.
2. Created `tests/test_preprocessing.py` with 5 unit tests validating label filtering, temporal extraction, stratified split balance, and imputer isolation.
3. Created `scripts/run_eda.py` generating 7 high-resolution figures in `reports/figures/eda/` and comprehensive audit `reports/eda_report.md`.
4. Authored `notebooks/01_exploratory_data_analysis.ipynb` with interactive visual cells.
5. Generated and persisted clean train/val/test splits to `data/processed/`.
6. Verified complete test suite passing (17/17 tests).
7. Updated `docs/IMPLEMENTATION_LOG.md`, `docs/AI_PROMPT_LOG.md`, and `docs/AI_FAILURE_LOG.md`.

---

## Stage 5: Target Definition, Leakage Audit & Data Splitting Strategy

### Exact User Prompt Received:
```text
Implement a rigorous target-definition and data-splitting
stage for AirSense.

Inspect the actual dataset schema first.

Determine whether AQI and a documented AQI category are available.

If an AQI-category column exists:
- Validate its labels.
- Identify missing and inconsistent labels.
- Preserve the documented categories.
- Explain their provenance.

If AQI exists but its category is absent:
- Investigate whether categories can be derived using an
  appropriate documented standard.
- Verify required pollutants, units, averaging periods,
  sub-indices, and applicable thresholds.
- Cite the selected official standard.
- Do not invent thresholds.

If neither a valid target nor enough information to derive one
exists, stop and report what is missing.

Create:
- src/target_definition.py
- src/data_split.py
- reports/target_definition.md
- tests/test_target_definition.py

Define the task as classification of a documented AQI category
from available input features.

CRITICAL LEAKAGE CHECK:
If the target AQI category is calculated from AQI, do not include
AQI itself as an input.

If the category is directly calculated from pollutant values,
explain that the classifier may reproduce the category rules.
Do not present this as proof of forecasting future pollution.

Check other possible leakage sources, including columns that
directly encode the target or measurements unavailable at
prediction time.

Use a reproducible split strategy. If timestamps are available,
consider chronological evaluation. If repeated cities or stations
exist, consider whether grouped evaluation is appropriate.

Document the chosen split, its justification, and limitations.

Run tests and wait for my review before moving forward.
```

### Objectives Identified & Executed:
1. Verified presence of both `AQI` and `AQI_Bucket` in `city_day.csv` ($N=29,531$). Validated all labels against official CPCB 2014 standards (0 mismatches across 24,850 rows).
2. Built `src/target_definition.py` implementing CPCB threshold derivation, target validation, and automated leakage detection/purging.
3. Built `src/data_split.py` supporting Stratified Split (primary benchmark, 70/15/15), Chronological Split, and Grouped Spatial Split.
4. Authored `reports/target_definition.md` detailing regulatory provenance, leakage safeguards, and comparative evaluation of splitting strategies.
5. Authored `tests/test_target_definition.py` and verified full test suite (24/24 tests passed).
6. Saved partitions to `data/processed/stratified/`, `temporal/`, and `spatial/`.
7. Updated `docs/IMPLEMENTATION_LOG.md`, `docs/AI_PROMPT_LOG.md`, and `docs/AI_FAILURE_LOG.md`.

---

## Support Entry: IDE Language Server Interpreter Mismatch Fix

### Exact User Prompt Received:
```text
Explain what this problem is and help me fix it: Cannot find module `pytest`
  Looked in these locations:
  Fallback search path (guessed from importing file with heuristics): ["c:\\dev\\AirSense", "c:\\dev\\AirSense\\tests", "c:\\dev\\AirSense", "c:\\dev", "c:\\"]
  Site packages path queried from interpreter: ["C:\\Program Files\\WindowsApps\\PythonSoftwareFoundation.Python.3.13_3.13.3824.0_x64__qbz5n2kfra8p0\\DLLs", "C:\\Program Files\\WindowsApps\\PythonSoftwareFoundation.Python.3.13_3.13.3824.0_x64__qbz5n2kfra8p0", "C:\\Users\\Admin\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python313\\site-packages", "C:\\Users\\Admin\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python313\\site-packages\\win32", "C:\\Users\\Admin\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python313\\site-packages\\win32\\lib", "C:\\Users\\Admin\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python313\\site-packages\\Pythonwin", "C:\\Program Files\\WindowsApps\\PythonSoftwareFoundation.Python.3.13_3.13.3824.0_x64__qbz5n2kfra8p0\\Lib\\site-packages"] @[c:\dev\AirSense\tests\test_target_definition.py:L10] 
```

### Objectives Identified & Executed:
1. Analyzed the error paths: revealed that the IDE language server was querying the Windows Store Python 3.13 environment (`PythonSoftwareFoundation.Python.3.13...`) where `pytest` is not installed.
2. Verified that the AirSense project dependencies are installed in Python 3.11 (`C:\Users\Admin\AppData\Local\Programs\Python\Python311\python.exe`), which has `pytest 8.3.5`.
3. Created `.vscode/settings.json` configuring `python.defaultInterpreterPath` to point to Python 3.11.
4. Documented the resolution for the user.

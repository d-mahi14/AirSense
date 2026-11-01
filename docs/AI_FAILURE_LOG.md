# AirSense — AI Failure & Error Log

This log records any errors, warnings, unexpected dataset characteristics, tool failures, and the exact resolutions applied during project development.

---

## Log Entry 1: Stage 1 — Inception & Workspace Inspection
- **Timestamp:** 2026-10-29T20:27:13+05:30 to 2026-10-29T20:35:00+05:30

### Issue 1: Git Repository Not Initialized
- **Description:** Executing `git status` returned exit code 1 (`fatal: not a git repository (or any of the parent directories): .git`).
- **Severity:** Informational / Constraint Verification.
- **Root Cause:** The working directory `C:\dev\AirSense` has not been initialized as a Git repository.
- **Resolution / Handling:** In accordance with the prompt's global rule (*"Use local Git commits only if Git is already configured and the working tree is safe"*), no `git init` was forced, and no local commits were created. All files are cleanly written to the local filesystem.

---

### Issue 2: PowerShell Quote Escaping SyntaxError in Python CLI Call
- **Description:** Running `python -c "print(f'{pkg}: {getattr(m, \"__version__\", \"installed\")}')"` resulted in `SyntaxError: unterminated string literal`.
- **Severity:** Low (CLI formatting error).
- **Root Cause:** Windows PowerShell processes backslash-escaped quotes differently inside outer double quotes, truncating the string passed to Python.
- **Resolution:** Re-wrote Python one-liners avoiding nested escaped double quotes (using single quotes or simple string concatenation), which executed with exit code 0.

---

### Issue 3: Missing Required Packages in Python 3.11 Environment
- **Description:** In the active Python 3.11 environment, `numpy`, `pandas`, `scipy`, `matplotlib`, and `seaborn` are installed, but `scikit-learn`, `xgboost`, `shap`, and `streamlit` are not installed.
- **Severity:** Medium (Action required before Stage 4/6 execution).
- **Resolution:** Detailed the package audit in `REQUIREMENTS.md` and generated `requirements.txt`. Flagged clearly for the user review before advancing to modeling or dashboard execution.

---

### Issue 4: Mixed Dtype Warning in `station_hour.csv`
- **Description:** Reading `station_hour.csv` produced `DtypeWarning: Columns (15) have mixed types. Specify dtype option on import or set low_memory=False.`
- **Severity:** Low (Data typing warning).
- **Root Cause:** Column 15 (`AQI_Bucket`) has mixed types across 2.58 million rows due to string labels intermixed with missing float NaNs.
- **Resolution:** Noted in `reports/dataset_requirements.md` that high-volume hourly reads must explicitly specify `dtype={'AQI_Bucket': str}` and `low_memory=False`.

---

### Issue 5: High Missingness in Volatile Organic Compounds (`Xylene`) and `PM10`
- **Description:** `Xylene` is missing in **61.32%** of `city_day.csv` and **78.81%** of `station_day.csv`. `PM10` is missing in **37.72%** of `city_day.csv`.
- **Severity:** Medium (Data quality challenge).
- **Root Cause:** Many continuous ambient air monitoring stations (CAAQMS) in India do not have active BTX (Benzene, Toluene, Xylene) or PM10 analyzers installed or operational.
- **Resolution:** Preprocessing architecture in `PROJECT_PLAN.md` includes grouped spatial-temporal median imputation and an explicit feature ablation experiment to determine whether excluding `Xylene` improves classification robustness.


---

## Log Entry 2: Stage 2 — Dataset Profiler & Data Loader Development
- **Timestamp:** 2026-10-29T20:40:18+05:30 to 2026-10-29T20:48:30+05:30

### Issue 6: Pytest Import Error (`ModuleNotFoundError: No module named 'src'`)
- **Description:** Running `pytest tests/test_data_loader.py` failed during collection with `ModuleNotFoundError: No module named 'src'`.
- **Severity:** Low (Test configuration).
- **Root Cause:** When running pytest directly without `-m pytest` or without an explicit root conftest, the project root directory was not prepended to Python's `sys.path`.
- **Resolution:** Added explicit `sys.path.insert(0, str(project_root))` inside `tests/test_data_loader.py` and standardized execution to `python -m pytest tests/test_data_loader.py -v`. All 12 tests executed cleanly.

---

### Issue 7: Custom Path Fallback in `get_data_dir()`
- **Description:** `test_get_data_dir_invalid_path` failed because passing an invalid `custom_path` fell back to existing default paths rather than failing.
- **Severity:** Low (Logic refinement).
- **Root Cause:** The initial fallback loop appended `custom_path` to standard candidate paths instead of strictly checking it first.
- **Resolution:** Refactored `get_data_dir(custom_path)` to strictly validate any explicitly provided custom path. If the path does not exist or lacks key CSVs, it immediately raises `FileNotFoundError`. If `custom_path` is None, it proceeds through standard discovery.

---

## Log Entry 3: Stage 3 — Theoretical Foundation & Literature Grounding
- **Timestamp:** 2026-10-29T20:51:52+05:30 to 2026-10-29T20:58:00+05:30

### Issue / Observation: Source Access Boundaries for Survey Paper
- **Description:** Access to the full paywalled PDF text of Méndez et al. (2023) directly from Springer Nature is restricted by standard institutional access boundaries.
- **Severity:** Informational / Scientific Integrity.
- **Resolution / Handling:** Openly declared source limitations in `references/README.md` and `reports/literature_review.md`. Used authorized public metadata from Springer Nature, ACM Digital Library, NIH/PubMed, and Universidad Complutense de Madrid to verify all key quantitative findings (155 papers surveyed over 2011–2021, taxonomy of algorithms, meteorological vs. pollutant variables, metric standards). No unsupported claims, fabricated page numbers, or phantom citations were introduced.

### Tool & Code Status:
- Zero syntax errors or tool failures during document compilation.
- Regression check on data loader tests: 12 passed in 1.08s.

---

## Log Entry 4: Stage 4 — Preprocessing Pipeline & Exploratory Data Analysis
- **Timestamp:** 2026-10-29T21:01:09+05:30 to 2026-10-29T21:08:00+05:30

### Issue 8: PowerShell Syntax Error with Nested Braces in Inline Notebook Script
- **Description:** Attempting to execute `python -c "..."` containing JSON dictionary braces `{}` and `%` operators caused PowerShell CLI parser errors.
- **Severity:** Low (CLI formatting).
- **Root Cause:** PowerShell interprets nested braces and `%` as script block delimiters and modulo/filter pipeline operators when unescaped.
- **Resolution:** Encapsulated the notebook generation logic into `scripts/generate_notebook.py` and executed via `python scripts/generate_notebook.py`. Notebook was successfully generated and verified.

### Issue 9: Matplotlib Font Fallback & Seaborn Palette Warning
- **Description:** Matplotlib emitted `findfont: Failed to find font weight semibold for Arial, now using 700.` and Seaborn emitted a deprecation warning regarding `palette` without `hue`.
- **Severity:** Low / Informational.
- **Resolution:** System automatically fell back to font weight 700. All 7 figures rendered cleanly and were written to `reports/figures/eda/` with zero missing visual artifacts.

---

## Log Entry 5: Stage 5 — Target Definition & Data Splitting
- **Timestamp:** 2026-10-29T21:11:40+05:30 to 2026-10-29T21:16:30+05:30

### Issue 10: PowerShell Quote Parsing in Python One-Liner
- **Description:** Running a quick validation one-liner with nested `f'\"{val}\"'` caused PowerShell syntax error `An expression was expected after '('`.
- **Severity:** Low (CLI formatting).
- **Resolution:** Re-wrote Python one-liner avoiding nested string escapes, executing with exit code 0.

### Tool & Code Status:
- Zero test failures across 24 unit tests in `tests/test_data_loader.py`, `tests/test_preprocessing.py`, and `tests/test_target_definition.py`.
- Automated leakage prevention verified: `prepare_feature_target_matrices()` confirmed to exclude continuous `AQI` and all target-derived columns.

---

## Log Entry 6: IDE Language Server Python Version Mismatch
- **Timestamp:** 2026-10-29T21:23:00+05:30 to 2026-10-29T21:25:00+05:30

### Issue 11: Editor Linter Error `Cannot find module pytest`
- **Description:** Pylance / Pyright linter in the editor flagged `import pytest` as unresolved in `tests/test_target_definition.py`.
- **Severity:** Low (Editor Language Server configuration).
- **Root Cause:** Windows has two Python installations:
  1. Python 3.11 (`C:\Users\Admin\AppData\Local\Programs\Python\Python311\python.exe`) with all project dependencies (`pytest 8.3.5`, `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`).
  2. Windows Store Python 3.13 (`C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13...`) with zero third-party packages.
  The editor automatically picked Python 3.13, causing import resolution warnings.
- **Resolution:** Added `.vscode/settings.json` specifying `"python.defaultInterpreterPath": "C:\\Users\\Admin\\AppData\\Local\\Programs\\Python\\Python311\\python.exe"`.


---

## Prompt 15: Audit findings and unresolved limits (2026-11-01T20:41:59.815133+05:30)

1. **Legacy runner bypassed the corrected protocol.** Its old module executed the
   random-split experiment at import time, set a misleading data-directory override
   and could overwrite current models. Original preserved; active file now has a
   main guard and calls corrected stages in order. Import safety/order tests pass.
2. **Legacy result disagreement.** models/all_results.json and model_results.md
   described older scores. Originals preserved. Compatibility export now derives
   from actual current outputs; the report is labelled historical. No submitted
   evaluation values or timing measurements were changed to look better.
3. **Split-audit documentation exceeded behavior.** The generic helper claimed
   overlap checks but omitted them. It now audits City-Date identities; an intentional
   reset-index overlap test demonstrates detection. The main chronological experiment
   already had disjoint identities and did not require changing its results.
4. **Folder spelling mismatch.** User requested archive(3); supplied archive (3).
   Both resolve and are tested/documented. No raw data was moved, copied or edited.
5. **Remaining environment limits.** Windows execution and browser visual layout
   are not verified here. AppTest and live HTTP checks passed; no browser-layout
   pass is invented. Paper authorship/affiliation still needs user confirmation.
6. **Scientific limits remain.** Historical test exposure cannot be undone by code
   fixes. Mixed city/time shifts, rare-class support, missingness/units/extremes,
   calendar clipping, absent weather and uncalibrated probabilities remain disclosed.

Final executed suite: 164 passed in 20.32s. No observed final test or
artifact-check failure is concealed. Full tuning/TreeSHAP was not rerun, and those
unexecuted repetitions are not recorded as passes. No further changes after handoff
without user review.


## Windows follow-up correction (1 November 2026)

User Windows run: 161 passed, 3 failed. Matching pinned versions ruled out a
simple dependency mismatch. Added native XGBoost JSON bundles and a shared
checksum-verified loader, without retraining or altering model scores. See
reports/windows_portability_fix.md (or windows_portability_fix.md within reports).
Earlier 164-test counts are historical Linux results; Windows confirmation of
this correction remains pending.

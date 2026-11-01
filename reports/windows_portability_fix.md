# Windows XGBoost portability correction — 1 November 2026

The user executed the previous package on Windows/Python 3.12.10 with the pinned
XGBoost 3.4.1, scikit-learn 1.8.0 and joblib 1.5.3. Actual result: 161 passed,
3 failed in 88.88s. Both original XGBoost memory snapshots failed to deserialize
with "input stream corrupted"; the prediction-form test also failed.
The previous Linux pass did not establish Windows compatibility.

The correction exports each original fitted XGBoost classifier with save_model
into native JSON. A separate joblib contains the exact fitted preprocessing and
sklearn wrapper with its booster snapshot removed. The loader reconstructs the
booster through load_model, preserves original wrapper parameters, and verifies
source/component hashes. It never retrains or changes the original models.
Training now emits these bundles; dashboard, evaluation, feature analysis and
audit share the loader. Unrelated legacy model utilities are unchanged.

Actual Linux checks: all eight candidates reproduce their saved validation/test
predictions and the deep feature/artifact audit passes. Both XGBoost portable
pipelines also produce bit-identical predictions AND probabilities to their
original pipelines on all validation and test observations. Regression tests
forbid loading the original snapshot and reject corrupted bundle components.
No tests were skipped to hide the Windows failure. Original metrics/timings and
model binaries remain unchanged. A live Windows rerun is still required; the
serialization explanation is supported by the error path, not proven by execution
on Windows in this environment.

Primary technical reference: XGBoost, Introduction to Model IO,
https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html
(verified 1 November 2026): native save_model/load_model is recommended for
stable storage; pickle memory snapshots are not a stable serialization contract.

## Apply the small patch

Stop Streamlit. Back up the project folder first. Extract AirSense_Windows_Fix.zip
into the existing AirSense project root, replacing matching supplied files.
Retain models/xgboost.joblib and models/tuned_xgboost.joblib: source hashes bind
the new companions to these originals. The updated full ZIP also includes the fix.

Run in CMD:

```bat
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m scripts.final_audit --deep-features
.venv\Scripts\python.exe -m streamlit run app.py
```

Do not push until the Windows rerun succeeds. All native JSON and portable joblib
companions must be committed alongside the original joblib models (the latter
are ignored by the current .gitignore and require explicit staging).

Latest full-suite outcome: **168 passed in 26.06s**, exit 0, Linux.
Command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q`.
Deep artifact command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.final_audit --deep-features`, exit 0.

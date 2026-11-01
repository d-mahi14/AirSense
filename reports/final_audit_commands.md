# Exact commands and actual outcomes

Recorded 2026-11-01T20:41:59.815133+05:30. Executed from `/workspace/scratch/c393465d1ff6/review/AirSense` in the local Linux runtime. The PowerShell commands in README are instructions for the target Windows machine and were not executed on Windows.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u -m scripts.final_audit --verify-training --deep-features
```

Exit 0. All eight saved test/validation predictions reproduced; all four baselines trained in temporary storage; submitted models unchanged; importance checks passed. Log: `final_audit_logs/artifact_and_isolated_training.log`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u -m scripts.final_audit --deep-features
```

Exit 0. Final source version: original raw hashes, training-only transform statistics, saved parameters/CV, model predictions, comparisons and feature outputs verified. Log: `final_audit_logs/artifact_deep_checks.log`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
```

Exit 0. 164 passed in 20.32s Log: `final_audit_logs/pytest.log`.

```bash
python -m pip check
```

Exit 0. No broken requirements found. Log: `final_audit_logs/pip_check.log`.

```bash
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python -m streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8502 --browser.gatherUsageStats false
```

Harness exit 0; server termination exit status was not retained. Started by a Python subprocess harness: health HTTP 200, body ok; app HTTP 200 with HTML. Server intentionally terminated after the checks. Log: `final_audit_logs/streamlit_server.log`.

Baseline training was reproduced through the isolated audit command, not by overwriting `models/` with the normal training CLI. Full hyperparameter search and full TreeSHAP computation were not rerun during prompt 15; existing CV records, fitted parameters, saved SHAP arrays, actual predictions and recomputed permutation outputs were verified instead. This keeps the original experiment and timing measurements intact. No Git or publication/deployment commands were executed.


## Windows follow-up correction (1 November 2026)

User Windows run: 161 passed, 3 failed. Matching pinned versions ruled out a
simple dependency mismatch. Added native XGBoost JSON bundles and a shared
checksum-verified loader, without retraining or altering model scores. See
reports/windows_portability_fix.md (or windows_portability_fix.md within reports).
Earlier 164-test counts are historical Linux results; Windows confirmation of
this correction remains pending.

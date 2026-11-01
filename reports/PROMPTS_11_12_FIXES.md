# Prompt 11 and 12 corrections

Only evaluation, tuning, feature analysis and their supporting artifacts were changed.
The dashboard and other prompt code remain as supplied.

- Training/tuning never evaluate test rows. Validation macro F1 freezes the selected artifact before test evaluation.
- Random splitting is replaced with shared chronological train/validation/test partitions. Temporal CV uses whole-date blocks and a seven-day gap, with fold-local preprocessing.
- All four models use training-data CV; six seeded candidates each limit cost. SVM searches 6,000 rows and refits on all 16,645 training rows.
- Timings cover the full pipeline for every model. Search and refit times are separate.
- All metrics, plots, per-class results, confusion analyses and importance findings derive from actual outputs. Failures are recorded explicitly.
- Feature-analysis sampling sizes match the code and manifests; XGBoost SHAP completed.
- Survey context uses verified sections and avoids unsupported superiority or direct-comparison claims.
- Raw data and unrelated source files are preserved. Legacy dashboard scores are outside this change and may still show earlier results.

## Completed verification

122 tests passed. Four baselines and four tuned finalists trained and evaluated successfully. Saved predictions independently reproduce every aggregate metric, per-class report and confusion matrix; all eight candidates used the same 4,421 test rows.

Random Forest baseline was selected using validation macro F1 = 0.774868. On the retrospective chronological test it achieved accuracy 0.807736 and macro F1 0.782597. XGBoost has slightly higher test accuracy (0.809545) but lower macro F1 (0.769019), illustrating why equal-class evaluation matters.

The split protocol changed, so these scores are not a like-for-like comparison with earlier random-split scores. Historical test exposure is disclosed; fresh external data is still needed for independent confirmation.

Reproduce using the four commands in EXPERIMENT_README.md. Detailed outputs are in model_comparison.md, hyperparameter_tuning.md, feature_analysis.md and verification/.

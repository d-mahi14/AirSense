# AirSense — corrected prompts 11 and 12

Install `requirements.txt` (or `requirements-dev.txt` for tests). The exact package
versions used for the delivered run are recorded in `requirements-experiment-lock.txt`.
Run these module commands from the AirSense project root:

```bash
python -m scripts.train_all_models
python -m scripts.tune_models
python -m scripts.evaluate_all_models
python -m scripts.analyze_features
python -m pytest -q
```

The older `run_all_models.py` and dashboard are outside this change; use the four
module commands above for the corrected prompt 11/12 experiment.
No `resume_tuning` source file is required or referenced.

## Experimental contract

- Shared chronological partitions: training through 30 June 2019, validation
  1 July–31 December 2019, test 1 January–1 July 2020. Dates are fixed in advance,
  rather than chosen from test performance. All models receive the same test rows.
- This estimates same-day AQI categories from contemporaneous measurements; it
  does not establish future forecasting or unseen-city generalization.
- Training and tuning request training/validation partitions only. Search uses
  three expanding date windows with a seven-day gap. Same-day observations from
  all cities stay together. Preprocessing is fitted inside each CV fold.
- Each model searches six seeded configurations. SVM searches 6,000 stratified
  training observations to control cost, then refits on all training rows, as do
  the other models. XGBoost searches tree counts with CV rather than early stopping
  on the selection validation set. Search spaces and complete CV results are saved.
- The highest validation macro F1 among all successful baseline/tuned candidates
  selects the candidate; exact ties use the candidate identifier. A signed-by-hash
  manifest freezes the roster and selected artifact before evaluation. These hashes
  provide integrity checks, not cryptographic authentication.
- Test evaluation checks dataset, protocol, package versions, frozen model hashes,
  and validation scores before processing test labels. It never selects or refits.
- Each inference time covers preprocessing plus prediction on all test rows,
  using the median of three runs after one small warm-up. Tuning cost is separately
  reported from full-training refit time.
- Metrics always use the fixed six-class CPCB label universe, including classes
  absent from an evaluation subset. No unavailable model receives invented scores.
- CSVs, figures, report interpretations, predictions and feature rankings derive
  from real model outputs. Failures are retained in JSON manifests and reports.

## Historical exposure and limitations

The uploaded project and the earlier audit had already examined test outcomes.
Changing the split cannot undo this knowledge: the delivered rerun is a retrospective
benchmark. An independently collected external dataset is needed for a pristine
confirmatory evaluation. No test results drive the corrected code's tuning or
selection. The 2020 period includes pandemic effects and different class proportions.

The split/CV protocol intentionally changes the earlier random benchmark. Old scores
are superseded, not evidence of a performance regression in the software. Other
reports and dashboard code from the uploaded project are left unchanged; they may
still display legacy scores. Use the prompt 11/12 reports listed below as authoritative.

## Key outputs

- `models/train_manifest.json`, `models/tuning_manifest.json`,
  `models/selection_manifest.json`: provenance, real validation metrics, selected
  candidate, model/dataset hashes, settings and failures.
- `reports/model_comparison.csv`: four baseline test results.
- `reports/all_candidate_comparison.csv`: all eight baseline/tuned test results.
- `reports/per_class_comparison.csv`, `reports/test_predictions.csv`,
  `reports/evaluation_results.json`: detailed actual results and failures.
- `reports/tuning/`: full candidate CV results; `reports/hyperparameter_tuning.md`
  records search spaces, winning parameters, validation deltas and compute costs.
- `reports/feature_analysis/`: importance CSVs, actual sampling manifest, training
  correlations and saved SHAP values; interpretation is in `reports/feature_analysis.md`.
- `reports/verification/`: execution logs and relevant test output.

The original survey is contextualized using its verified predictor and metric
sections, with no claim of direct numerical comparability. Feature importance
is association, not causation. Weather fields are absent and city identity is excluded.

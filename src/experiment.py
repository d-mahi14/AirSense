"""Shared experiment contract for prompts 11 and 12.

Training/tuning never evaluate or transform test rows. Model selection uses validation only and
is frozen with data/model hashes before the separate final evaluation command.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
from src.model_io import load_pipeline, save_pipeline
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import get_data_dir
from src.feature_engineering import prepare_modeling_dataset, get_feature_lists, extract_X_y
from src.pipeline import build_preprocessing_pipeline
from src.evaluation import compute_metrics

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
NAMES = {"decision_tree": "Decision Tree", "random_forest": "Random Forest", "svm": "SVM", "xgboost": "XGBoost"}
TRAIN_END = "2019-06-30"
VAL_END = "2019-12-31"
PROTOCOL = "chronological-v2"
HISTORY = ("The uploaded project and the earlier audit had already inspected test outcomes. "
           "This corrected rerun is a retrospective benchmark, not a pristine prospective holdout. "
           "No test scores are used by the corrected selection/tuning code. New external data is "
           "required for an independent final claim.")
SURVEY = ("Méndez, Merayo & Núñez (2023), Machine learning algorithms to forecast air quality: "
          "a survey. https://doi.org/10.1007/s10462-023-10424-4 (publisher verified 2026-11-01). "
          "Section 3.3 discusses predictor groups, including pollutant and weather variables. "
          "Section 3.4 distinguishes metric types and reports widespread use of RMSE/MSE and MAE "
          "as well as ACC, R² and MAPE. AirSense instead evaluates six-class, same-day AQI "
          "classification using macro F1 and per-class errors. Chronological splitting and "
          "macro-F1 selection here are our experimental choices, not attributed recommendations "
          "from the survey. Datasets, targets, horizons, splits and metrics differ, so our scores "
          "are not directly comparable with published scores. No universal ensemble advantage "
          "or macro-F1 recommendation is attributed to this survey.")


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_default, allow_nan=False), encoding="utf-8")


def _json_default(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    raise TypeError(type(v).__name__)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_path():
    return get_data_dir() / "city_day.csv"


def data_for(split):
    """Load just the requested date partition, then drop unlabeled records.

    Training and validation calls do not encode, transform or return test rows.
    CSV byte hashing is permitted solely for provenance, not model decisions.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError(split)
    df = pd.read_csv(source_path(), parse_dates=["Date"])
    if split == "train":
        mask = df.Date <= TRAIN_END
    elif split == "validation":
        mask = (df.Date > TRAIN_END) & (df.Date <= VAL_END)
    else:
        mask = df.Date > VAL_END
    df = df.loc[mask & df.AQI_Bucket.notna()].sort_values(["Date", "City"]).reset_index(drop=True)
    if df.duplicated(["City", "Date"]).any():
        raise ValueError("Duplicate city/date identities in source")
    prepared = prepare_modeling_dataset(df, include_city=False)
    features = get_feature_lists(prepared, include_city=False)
    X, y = extract_X_y(prepared, features)
    return df, X, y, features


def identity():
    import sklearn, xgboost
    return {"protocol": PROTOCOL, "dataset_sha256": sha(source_path()), "train_end": TRAIN_END,
            "validation_end": VAL_END, "seed": 42, "python": platform.python_version(),
            "sklearn": sklearn.__version__, "xgboost": xgboost.__version__}


def make_pipeline(key, features):
    from sklearn.pipeline import Pipeline
    pre = build_preprocessing_pipeline(features, scale_numerical=True, winsorize=True)
    common = {"random_state": 42}
    if key == "decision_tree":
        clf = DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=10,
                                     class_weight="balanced", **common)
    elif key == "random_forest":
        clf = RandomForestClassifier(n_estimators=200, min_samples_split=10, min_samples_leaf=5,
                                     class_weight="balanced", n_jobs=2, **common)
    elif key == "svm":
        clf = SVC(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced", **common)
    elif key == "xgboost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                            objective="multi:softprob", num_class=6, eval_metric="mlogloss",
                            tree_method="hist", n_jobs=2, **common)
    else:
        raise ValueError(key)
    return Pipeline(pre.steps + [("classifier", clf)])


def prediction_metrics(pipeline, X, y, split):
    # One warm-up followed by three complete preprocessing + prediction calls.
    pipeline.predict(X.iloc[:min(32, len(X))])
    elapsed = []
    prediction = None
    for _ in range(3):
        start = time.perf_counter()
        current = pipeline.predict(X)
        elapsed.append(time.perf_counter() - start)
        if prediction is not None and not np.array_equal(current, prediction):
            raise ValueError("Non-deterministic repeated predictions")
        prediction = current
    metrics = compute_metrics(y, prediction, split, float(np.median(elapsed)))
    metrics["inference_runs_sec"] = elapsed
    metrics["inference_ms_per_row"] = metrics["inference_time_sec"] * 1000 / len(X)
    return metrics, prediction


def candidate_id(key, tuned=False):
    return ("tuned_" if tuned else "") + key


def choose_validation(results):
    valid = [(key, val) for key, val in results.items() if "error" not in val]
    if not valid:
        raise ValueError("No successful validation candidates")
    # Lexicographic identifier breaks exact ties deterministically, without test access.
    return sorted(valid, key=lambda item: (-item[1]["validation_metrics"]["macro_f1"], item[0]))[0][0]


def freeze_selection(candidates, experiment):
    selected = choose_validation(candidates)
    manifest = {
        "experiment": experiment, "created_utc": datetime.now(timezone.utc).isoformat(),
        "criterion": "Highest validation macro F1 across baseline and tuned candidates; identifier breaks ties",
        "selected_candidate": selected,
        "candidates": {k: {"validation_macro_f1": v["validation_metrics"]["macro_f1"],
                            "model_sha256": v["model_sha256"]}
                       for k, v in candidates.items() if "error" not in v},
        "historical_test_exposure": HISTORY,
    }
    dump(MODELS / "selection_manifest.json", manifest)
    return manifest


def fit_baselines():
    MODELS.mkdir(exist_ok=True)
    experiment = identity()
    _, X, y, features = data_for("train")
    _, Xv, yv, _ = data_for("validation")
    results = {}
    # Old final results must not masquerade as results from a new run.
    for name in ["selection_manifest.json", "tuning_manifest.json"]:
        (MODELS / name).unlink(missing_ok=True)
    for key in NAMES:
        name = candidate_id(key)
        path = MODELS / f"{name}.joblib"
        path.unlink(missing_ok=True)
        (MODELS / f"{name}_config.json").unlink(missing_ok=True)
        try:
            pipe = make_pipeline(key, features)
            start = time.perf_counter()
            pipe.fit(X, y)
            elapsed = time.perf_counter() - start
            metrics, _ = prediction_metrics(pipe, Xv, yv, "Validation")
            save_pipeline(pipe, path)
            result = {"model": NAMES[key], "variant": "baseline", "train_time_sec": elapsed,
                      "n_train": len(X), "n_val": len(Xv), "used_n_train": len(X),
                      "hyperparameters": {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                                          for k, v in pipe.named_steps["classifier"].get_params().items()},
                      "val_macro_f1": metrics["macro_f1"], "val_accuracy": metrics["accuracy"],
                      "validation_metrics": metrics, "model_sha256": sha(path)}
            dump(MODELS / f"{name}_config.json", result)
            results[name] = result
            print(name, "validation macro F1", metrics["macro_f1"], flush=True)
        except Exception as exc:
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(name, results[name], flush=True)
    dump(MODELS / "train_manifest.json", {"experiment": experiment, "candidates": results})
    freeze_selection(results, experiment)
    return results


def validate_manifest(manifest):
    if manifest["experiment"] != identity():
        raise ValueError("Data/protocol/environment changed; rerun training and tuning")
    for name, result in manifest["candidates"].items():
        if "error" not in result and sha(MODELS / f"{name}.joblib") != result["model_sha256"]:
            raise ValueError(f"Model hash changed: {name}")


def tune_all():
    from src.tuning import get_search_spaces, run_search
    train = read(MODELS / "train_manifest.json")
    validate_manifest(train)
    df, X, y, features = data_for("train")
    _, Xv, yv, _ = data_for("validation")
    results = {}
    spaces = get_search_spaces()
    search_dir = REPORTS / "tuning"
    search_dir.mkdir(parents=True, exist_ok=True)
    for key in NAMES:
        name = candidate_id(key, True)
        path = MODELS / f"{name}.joblib"
        path.unlink(missing_ok=True)
        (MODELS / f"{name}_config.json").unlink(missing_ok=True)
        try:
            base = make_pipeline(key, features)
            # Avoid nested forest/XGBoost thread pools during outer CV parallelism.
            if "classifier__n_jobs" in base.get_params():
                base.set_params(classifier__n_jobs=1)
            indices = np.arange(len(X))
            if key == "svm" and len(X) > 6000:
                from sklearn.model_selection import train_test_split
                indices, _ = train_test_split(indices, train_size=6000, stratify=y, random_state=42)
                indices = np.sort(indices)
            Xs, ys = X.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True)
            dates = df.Date.iloc[indices].reset_index(drop=True)
            params, cv, search_time = run_search(base, spaces[key], Xs, ys, dates)
            cv.to_csv(search_dir / f"{key}_cv_results.csv", index=False)
            best_cv = float(cv.mean_test_score.max())
            pipe = make_pipeline(key, features).set_params(**params)
            start = time.perf_counter()
            pipe.fit(X, y)  # ALL tuned finalists use full training data, including SVM.
            refit_time = time.perf_counter() - start
            metrics, _ = prediction_metrics(pipe, Xv, yv, "Validation")
            save_pipeline(pipe, path)
            result = {"model": NAMES[key], "variant": "tuned", "best_params": params,
                      "search_space": spaces[key], "cv_best_f1": best_cv,
                      "cv_method": "3 expanding date windows, 7-day gap, preprocessing within folds",
                      "search_rows": len(Xs), "n_train": len(X), "used_n_train": len(X),
                      "search_time_sec": search_time, "train_time_sec": refit_time,
                      "val_macro_f1": metrics["macro_f1"], "val_accuracy": metrics["accuracy"],
                      "validation_metrics": metrics, "model_sha256": sha(path)}
            dump(MODELS / f"{name}_config.json", result)
            results[name] = result
            print(name, "validation macro F1", metrics["macro_f1"], flush=True)
        except Exception as exc:
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(name, results[name], flush=True)
        # Persist partial progress and explicit failures so interrupted runs are auditable.
        dump(MODELS / "tuning_manifest.json", {"experiment": train["experiment"], "candidates": results})
    candidates = {**train["candidates"], **results}
    selection = freeze_selection(candidates, train["experiment"])
    write_tuning_report(train["candidates"], results, selection)
    return results


def markdown_table(frame):
    # Avoid an undeclared optional tabulate dependency.
    def fmt(v):
        return f"{v:.6f}" if isinstance(v, (float, np.floating)) else str(v)
    return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |",
                       "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
                      ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)])


def write_tuning_report(baselines, tuned, selection):
    lines = ["# AirSense — Hyperparameter tuning", "", HISTORY, "",
             "## Protocol", "", "Training: dates through 2019-06-30. Validation: 2019-07-01 through 2019-12-31. "
             "Test is evaluated only by the separate final evaluation command. Search uses training rows only, "
             "three expanding date windows with a seven-day gap, and fold-local preprocessing. "
             "All cities on one date stay on the same side of a fold. This targets future dates in observed cities, "
             "not new-city transfer. Six seeded random candidates per model limit computation. "
             "XGBoost uses CV with fixed tree-count candidates; no validation early stopping. "
             "SVM search is limited to 6,000 stratified training rows; its finalist is refitted on every training row.",
             "", "## Baseline versus tuned — validation only", ""]
    rows = []
    for key in NAMES:
        b, t = baselines.get(key, {}), tuned.get("tuned_" + key, {})
        if "validation_metrics" in b and "validation_metrics" in t:
            bm, tm = b["validation_metrics"], t["validation_metrics"]
            rows.append({"model": NAMES[key], "baseline_macro_f1": bm["macro_f1"],
                         "tuned_macro_f1": tm["macro_f1"], "delta_f1": tm["macro_f1"] - bm["macro_f1"],
                         "baseline_accuracy": bm["accuracy"], "tuned_accuracy": tm["accuracy"]})
    if rows:
        table = pd.DataFrame(rows)
        table.to_csv(REPORTS / "tuning_validation_comparison.csv", index=False)
        lines += [markdown_table(table), ""]
    for name, result in tuned.items():
        lines += [f"## {name}", "", "```json", json.dumps(result, indent=2, default=_json_default), "```", ""]
    lines += ["## Selection and limits", "", f"Frozen candidate: **{selection['selected_candidate']}**, chosen by "
              "validation macro F1 across all successful baseline and tuned candidates. Negative deltas are retained; "
              "tuning is not guaranteed to help. One seasonally limited validation period and a small search budget "
              "make selection uncertain. CV scores on the SVM subset are not directly comparable to full-data CV "
              "scores. Within-city autocorrelation may persist beyond seven days. No city-disjoint generalization "
              "claim is made. A single seed is used; uncertainty across splits/seeds is not measured.", "",
              "## Research context", "", SURVEY, "", "## Reproduce", "",
              "`python -m scripts.train_all_models` then `python -m scripts.tune_models`, "
              "then `python -m scripts.evaluate_all_models`. Full CV results are in `reports/tuning/`; "
              "search spaces, best parameters, actual row counts, failures and timing are stored above and in manifests."]
    (REPORTS / "hyperparameter_tuning.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate_all():
    from src.evaluation import (build_comparison_table, plot_performance_comparison,
                               plot_macro_f1_comparison, plot_confusion_matrix, plot_per_class_performance)
    selection = read(MODELS / "selection_manifest.json")
    if selection["experiment"] != identity():
        raise ValueError("Frozen selection does not match current dataset/protocol/environment")
    candidates = read(MODELS / "train_manifest.json")["candidates"]
    if (MODELS / "tuning_manifest.json").exists():
        tuning = read(MODELS / "tuning_manifest.json")
        if tuning["experiment"] != selection["experiment"]:
            raise ValueError("Tuning and selection experiment mismatch")
        candidates.update(tuning["candidates"])
    # Validate the frozen roster and every artifact BEFORE test labels are processed.
    valid_roster = {name for name, result in candidates.items() if "error" not in result}
    if valid_roster != set(selection["candidates"]):
        raise ValueError("Candidate roster changed since selection; rerun tuning/selection")
    if choose_validation(candidates) != selection["selected_candidate"]:
        raise ValueError("Selection is inconsistent with validation metrics")
    for name in valid_roster:
        record = selection["candidates"][name]
        if (record["model_sha256"] != candidates[name]["model_sha256"] or
            sha(MODELS / f"{name}.joblib") != record["model_sha256"] or
            record["validation_macro_f1"] != candidates[name]["validation_metrics"]["macro_f1"]):
            raise ValueError(f"Model/validation record changed since selection: {name}")
    frozen_sha = sha(MODELS / "selection_manifest.json")
    df, X, y, _ = data_for("test")
    results, predictions, per_class = {}, [], []
    for name, candidate in candidates.items():
        if "error" in candidate:
            results[name] = candidate
            continue
        try:
            pipeline = load_pipeline(MODELS / f"{name}.joblib")
            metrics, pred = prediction_metrics(pipeline, X, y, "Test")
            results[name] = {**candidate, "test_metrics": metrics, "val_metrics": candidate["validation_metrics"]}
            pred_frame = df[["City", "Date"]].copy()
            pred_frame["candidate"] = name
            pred_frame["y_true"] = np.asarray(y)
            pred_frame["y_pred"] = pred
            predictions.append(pred_frame)
            for label, value in metrics["per_class_report"].items():
                if label in metrics["class_labels"]:
                    per_class.append({"candidate": name, "class": label, **value})
            print(name, "test macro F1", metrics["macro_f1"], flush=True)
        except Exception as exc:
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
    if frozen_sha != sha(MODELS / "selection_manifest.json"):
        raise ValueError("Selection was modified during evaluation")
    REPORTS.mkdir(exist_ok=True)
    dump(REPORTS / "evaluation_results.json", {"experiment": selection["experiment"],
         "selection_manifest_sha256": frozen_sha, "selected_candidate": selection["selected_candidate"],
         "test_rows": len(X), "candidates": results})
    pd.concat(predictions, ignore_index=True).to_csv(REPORTS / "test_predictions.csv", index=False) if predictions else None
    pc = pd.DataFrame(per_class)
    pc.to_csv(REPORTS / "per_class_comparison.csv", index=False)
    # Primary CSV compares the four baselines; extended CSV includes all eight candidates.
    table = build_comparison_table(results, split="Test")
    table.to_csv(REPORTS / "all_candidate_comparison.csv", index=False)
    baseline_results = {NAMES[k]: results[k] for k in NAMES}
    baseline_table = build_comparison_table(baseline_results, split="Test")
    baseline_table.to_csv(REPORTS / "model_comparison.csv", index=False)
    fig = REPORTS / "figures" / "model_comparison"
    fig.mkdir(parents=True, exist_ok=True)
    # Remove stale images from older runs, including models that now failed.
    for old in fig.glob("*.png"):
        old.unlink()
    if not baseline_table.empty:
        plot_performance_comparison(baseline_table, fig)
        plot_macro_f1_comparison(baseline_table, baseline_results, fig)
        plot_per_class_performance(baseline_results, fig)
    for name, result in results.items():
        if "test_metrics" in result:
            m = result["test_metrics"]
            plot_confusion_matrix(m["confusion_matrix"], m["class_labels"], name, fig,
                                  filename_suffix="")
    write_evaluation_report(results, baseline_table, table, pc, selection, len(X))
    return results


def write_evaluation_report(results, baselines, table, pc, selection, n_test):
    lines = ["# AirSense — Unified model comparison", "", HISTORY, "",
             "## Experimental protocol", "",
             "Chronological shared partitions: train through 2019-06-30; validation July–December 2019; "
             f"final retrospective test January–July 2020 ({n_test:,} labeled city-days). "
             "All successful models receive the exact same ordered test rows. AQI and its bucket are excluded "
             "from predictors; the task is same-day category estimation from contemporaneous pollutants, "
             "not future forecasting. City identity is excluded. No weather fields are available.", "",
             f"**Selected candidate: {selection['selected_candidate']}**. {selection['criterion']}. "
             "The selection manifest and model hashes were checked before test evaluation; test scores "
             "are descriptive and do not change this choice.", "",
             "## Baseline models — actual test predictions", "", markdown_table(baselines), "",
             "## Baseline and tuned candidates — actual test predictions", "", markdown_table(table), "",
             "Training time includes fitting the full preprocessing/classifier pipeline. For tuned models it "
             "is the full-data refit time; search costs are separately recorded in the tuning report. "
             "Inference time is the median of three complete pipeline prediction calls after one small warm-up, "
             "on identical test rows, including preprocessing for every model. Hardware-dependent timings "
             "are not portable benchmarks.", "",
             "![Performance](figures/model_comparison/performance_comparison.png)", "",
             "![Macro F1](figures/model_comparison/macro_f1_comparison.png)", "",
             "![Per-class F1](figures/model_comparison/per_class_performance.png)", "",
             "## Per-class precision, recall and F1", "", markdown_table(pc), "",
             "## Accuracy versus macro F1", ""]
    if not baselines.empty:
        acc_order = baselines.sort_values("accuracy", ascending=False).model.tolist()
        f1_order = baselines.sort_values("macro_f1", ascending=False).model.tolist()
        lines += [f"Accuracy ranking: {', '.join(acc_order)}. Macro F1 ranking: {', '.join(f1_order)}. "
                  f"The rankings {'agree' if acc_order == f1_order else 'differ'}. "
                  "Agreement alone does not establish good minority-class performance; inspect per-class recall. "
                  "Macro F1 assigns equal weight to each of the six CPCB classes.", ""]
    if not pc.empty:
        means = pc.groupby("class")["f1-score"].mean().sort_values()
        lines += [f"Lowest mean class F1 across successful candidates: **{means.index[0]}** "
                  f"({means.iloc[0]:.4f}); highest: **{means.index[-1]}** ({means.iloc[-1]:.4f}).", ""]
    if not table.empty:
        row = table.sort_values("macro_f1", ascending=False).iloc[0]
        lines += [f"Highest observed test macro F1: **{row['model']}** ({row['macro_f1']:.4f}). "
                  "This retrospective ranking is distinct from the validation-selected deployment candidate.", ""]
    lines += ["## Confusion matrices and measured errors", "",
              "Rows are true labels and columns predictions; category order runs Good to Severe. "
              "Above-diagonal predictions overestimate severity; below-diagonal predictions underestimate it. "
              "No claim about errors near regulatory breakpoints is made without a separate distance-to-boundary analysis.", ""]
    for name, result in results.items():
        if "error" in result:
            lines += [f"**{name}: FAILED — {result['error']}**", ""]
            continue
        m = result["test_metrics"]
        cm = np.asarray(m["confusion_matrix"])
        wrong = int(cm.sum() - np.trace(cm))
        i, j = np.indices(cm.shape)
        adjacent = int(cm[(abs(i - j) == 1)].sum())
        off = cm.copy()
        np.fill_diagonal(off, 0)
        r, c = np.unravel_index(off.argmax(), off.shape)
        lines += [f"### {name}", "", f"Errors: {wrong}; adjacent-category errors: {adjacent} "
                  f"({adjacent / wrong:.1%} of errors)." if wrong else "No test errors.", "",
                  f"Largest off-diagonal count: {m['class_labels'][r]} → {m['class_labels'][c]}: {int(off[r,c])}. "
                  f"Underestimates: {int(cm[i > j].sum())}; overestimates: {int(cm[i < j].sum())}.", "",
                  f"![{name}](figures/model_comparison/confusion_matrix_{name}.png)", ""]
    lines += ["## Limitations", "",
              "One dataset, one seed and fixed chronological periods; no confidence intervals or statistical "
              "significance claims. The 2020 test period includes unusual pandemic conditions and seasonal/class "
              "distribution shift. Adjacent city-days remain dependent even with date-blocked CV. "
              "City-disjoint transfer has not been tested. Same-day pollutant inputs are related to the AQI "
              "definition, so high accuracy does not establish forecasting ability. Missingness, imputation, "
              "winsorisation and exclusion of unlabeled days may bias results. Weather and lead measurements "
              "are unavailable. Historical test exposure prevents an independent confirmatory claim.", "",
              "## Comparison with primary survey", "", SURVEY, "",
              "## Reproduce", "",
              "`python -m scripts.train_all_models` → `python -m scripts.tune_models` → "
              "`python -m scripts.evaluate_all_models` → `python -m scripts.analyze_features`. "
              "Machine-readable predictions, confusion matrices, failures and manifests are saved alongside this report."]
    (REPORTS / "model_comparison.md").write_text("\n".join(lines), encoding="utf-8")

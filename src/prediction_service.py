"""Cached, read-only inference using AirSense's saved training pipelines.

No training or tuning function is imported or called by this module. Input schema
comes from the selected fitted artifact; calendar fields reuse training code.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from datetime import date
from pathlib import Path
from typing import Mapping

import joblib
from src.model_io import load_pipeline
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

from src.feature_engineering import NUMERICAL_POLLUTANT_FEATURES, NUMERICAL_TEMPORAL_FEATURES
from src.preprocessing import extract_temporal_features, LABEL_TO_CATEGORY, SEASON_MAP

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_NAMES = {"decision_tree": "Decision Tree", "random_forest": "Random Forest", "svm": "SVM", "xgboost": "XGBoost"}
AQI_CATEGORIES = [LABEL_TO_CATEGORY[i] for i in range(6)]
AQI_COLORS = {"Good": "#21865c", "Satisfactory": "#5b8737", "Moderate": "#a17a16", "Poor": "#c16a24", "Very Poor": "#ba3e45", "Severe": "#783e88"}
POLLUTANT_COLS = list(NUMERICAL_POLLUTANT_FEATURES)
CALENDAR_COLS = list(NUMERICAL_TEMPORAL_FEATURES) + ["Season"]


class ArtifactError(ValueError):
    """A missing, stale or incompatible saved artifact."""


class InputValidationError(ValueError):
    """Inputs do not match the trained model contract."""


def signature(path):
    path = Path(path)
    try:
        stat = path.stat()
        return (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@st.cache_data(show_spinner=False)
def _read_json_cached(path: str, stamp):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value, None
    except (OSError, ValueError) as exc:
        return {}, f"{Path(path).name}: {exc}"


def read_json(path):
    return _read_json_cached(str(path), signature(path))


def load_evaluation():
    """Return current metrics only when their frozen selection still matches."""
    evaluation, error = read_json(REPORTS_DIR / "evaluation_results.json")
    if error:
        return {}, error
    selection_path = MODEL_DIR / "selection_manifest.json"
    if signature(selection_path) is None:
        return {}, "Selection manifest is missing; evaluation provenance cannot be verified."
    if evaluation.get("selection_manifest_sha256") != file_sha256(selection_path):
        return {}, "Evaluation results are stale relative to the selection manifest. Rerun final evaluation separately."
    if not isinstance(evaluation.get("candidates"), dict):
        return {}, "Evaluation results have no candidate records."
    # Validate labels and shapes before any UI treats the JSON as performance data.
    try:
        for record in evaluation["candidates"].values():
            if "error" in record:
                continue
            m = record["test_metrics"]
            cm = np.asarray(m["confusion_matrix"], dtype=float)
            if cm.shape != (6, 6) or not np.isfinite(cm).all() or (cm < 0).any():
                raise ValueError("Invalid confusion matrix")
            if m["class_labels"] != AQI_CATEGORIES or cm.sum() != m["n_samples"]:
                raise ValueError("Class labels/support do not match")
            for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
                if not np.isfinite(m[key]) or not 0 <= m[key] <= 1:
                    raise ValueError(f"Invalid {key}")
    except (KeyError, TypeError, ValueError) as exc:
        return {}, f"Invalid evaluation artifact: {exc}"
    return evaluation, None


def candidate_key(model_name, variant="baseline"):
    for key, display in MODEL_NAMES.items():
        if model_name == display:
            if variant not in {"baseline", "tuned"}:
                raise ValueError("Unknown model variant")
            return ("tuned_" if variant == "tuned" else "") + key
    raise ValueError("Unknown model name")


def candidate_label(candidate):
    tuned = candidate.startswith("tuned_")
    key = candidate.removeprefix("tuned_")
    return f"{MODEL_NAMES.get(key, key)} · {'tuned' if tuned else 'baseline'}"


def model_schema(pipeline):
    try:
        names = list(pipeline.feature_names_in_)
    except AttributeError as exc:
        raise ArtifactError("Saved pipeline does not expose its fitted input feature names.") from exc
    supported = set(POLLUTANT_COLS + CALENDAR_COLS)
    unknown = set(names) - supported
    if unknown or len(names) != len(set(names)):
        raise ArtifactError(f"Unsupported model input schema: {sorted(unknown)}")
    if not any(name in POLLUTANT_COLS for name in names):
        raise ArtifactError("The selected pipeline has no supported pollutant predictors.")
    if not hasattr(pipeline, "named_steps") or "classifier" not in pipeline.named_steps:
        raise ArtifactError("Expected a complete preprocessing/classifier pipeline.")
    return names


@st.cache_resource(show_spinner=False, max_entries=8)
def _load_model_cached(path: str, stamp, expected_sha: str | None, sklearn_version: str | None):
    try:
        if stamp is None:
            raise ArtifactError(f"Model file is missing: {Path(path).name}")
        if expected_sha and file_sha256(path) != expected_sha:
            raise ArtifactError("Model checksum differs from its evaluation record. Restore the matching artifact.")
        if sklearn_version:
            from sklearn import __version__
            if __version__ != sklearn_version:
                raise ArtifactError(f"Saved model requires scikit-learn {sklearn_version}; installed {__version__}. Install requirements.txt.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            pipeline = load_pipeline(path)
        model_schema(pipeline)
        return pipeline
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError(f"Could not load {Path(path).name}: {exc}. Install the pinned dependencies.") from exc


def load_model(candidate, evaluation=None):
    allowed = set(MODEL_NAMES) | {"tuned_" + name for name in MODEL_NAMES}
    if candidate not in allowed:
        raise ArtifactError("Unknown model identifier")
    evaluation = evaluation or {}
    record = evaluation.get("candidates", {}).get(candidate, {})
    if "error" in record:
        raise ArtifactError(f"This candidate failed during the experiment: {record['error']}")
    path = MODEL_DIR / f"{candidate}.joblib"
    return _load_model_cached(str(path), signature(path), record.get("model_sha256"),
                              evaluation.get("experiment", {}).get("sklearn"))


@st.cache_data(show_spinner=False)
def _dataset_cached(path: str, stamp):
    frame = pd.read_csv(path, parse_dates=["Date"])
    needed = set(["Date", "City", "AQI_Bucket"] + POLLUTANT_COLS)
    if not needed.issubset(frame.columns):
        raise ArtifactError("city_day.csv is missing expected dataset columns")
    return frame


def load_dataset():
    from src.data_loader import get_data_dir
    try:
        path = get_data_dir() / "city_day.csv"
        return _dataset_cached(str(path), signature(path))
    except Exception as exc:
        raise ArtifactError(f"Dataset unavailable: {exc}") from exc


def training_rows(dataset, train_end="2019-06-30"):
    return dataset.loc[(dataset.Date <= pd.Timestamp(train_end)) & dataset.AQI_Bucket.notna()].copy()


def input_ranges(training):
    """UI bounds are engineering limits, not AQI or sensor standards.

    Nonnegative to 120% of each observed training maximum, rounded upward.
    Medians and observed maxima are disclosed; no test-derived limits are used.
    """
    bounds = {}
    for feature in POLLUTANT_COLS:
        series = pd.to_numeric(training[feature], errors="coerce")
        series = series[np.isfinite(series) & (series >= 0)]
        if series.empty:
            raise ArtifactError(f"No finite training measurements for {feature}; cannot derive input bounds.")
        maximum = float(series.max())
        bounds[feature] = {"min": 0.0, "max": float(max(1, np.ceil(maximum * 1.2))),
                           "training_max": maximum, "median": float(series.median())}
    return bounds


def build_model_input(pipeline, pollutants: Mapping[str, object], measurement_date, bounds):
    names = model_schema(pipeline)
    required = [feature for feature in names if feature in POLLUTANT_COLS]
    extra = set(pollutants) - set(required)
    if extra:
        raise InputValidationError(f"Unexpected pollutant fields: {', '.join(sorted(extra))}")
    try:
        timestamp = pd.Timestamp(measurement_date)
        if pd.isna(timestamp) or timestamp.tzinfo is not None:
            raise ValueError("Invalid or timezone-aware date")
        if timestamp.date() > date.today() or timestamp.year < 2015:
            raise ValueError("Use an actual measurement date between 2015 and today")
        temporal = extract_temporal_features(pd.DataFrame({"Date": [timestamp]}))
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError(f"Invalid measurement date: {exc}") from exc
    values, missing, extrapolated = {}, [], []
    for feature in required:
        raw = pollutants.get(feature)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            values[feature] = np.nan
            missing.append(feature)
            continue
        try:
            if isinstance(raw, (bool, np.bool_)):
                raise ValueError("Boolean measurements are invalid")
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise InputValidationError(f"{feature} must be a number or blank.") from exc
        if not np.isfinite(number):
            raise InputValidationError(f"{feature} must be finite; use blank for an unavailable reading.")
        if feature not in bounds:
            raise InputValidationError(f"Permitted range for {feature} is unavailable.")
        limit = bounds[feature]
        if not limit["min"] <= number <= limit["max"]:
            raise InputValidationError(f"{feature} must be between {limit['min']:g} and {limit['max']:g} in source-dataset units.")
        if number > limit["training_max"]:
            extrapolated.append(feature)
        values[feature] = number
    if len(missing) == len(required):
        raise InputValidationError("Enter at least one pollutant measurement. All readings are blank.")
    # Training's own temporal extractor gives matching calendar values and dtypes.
    for feature in CALENDAR_COLS:
        if feature in names:
            values[feature] = temporal[feature].iloc[0]
    frame = pd.DataFrame({feature: pd.Series([values[feature]], dtype=(temporal[feature].dtype if feature in CALENDAR_COLS else 'float64'))
                          for feature in names}, columns=names)
    return frame, {"imputed_pollutants": missing, "above_training_max": extrapolated}


def predict(pipeline, input_frame):
    """Return category and label-aligned, uncalibrated class probabilities."""
    if len(input_frame) != 1 or list(input_frame.columns) != model_schema(pipeline):
        raise InputValidationError("Prediction requires one row in the fitted pipeline's exact feature order.")
    try:
        labels = np.asarray(pipeline.predict(input_frame))
        if labels.shape != (1,) or labels[0] not in LABEL_TO_CATEGORY:
            raise ArtifactError("Model returned an unknown AQI class")
        result = {"category": LABEL_TO_CATEGORY[labels[0]], "probabilities": None, "probability_note": None}
        if hasattr(pipeline, "predict_proba"):
            try:
                raw = np.asarray(pipeline.predict_proba(input_frame)[0], dtype=float)
                classes = np.asarray(pipeline.classes_)
                if raw.shape != classes.shape or not np.isfinite(raw).all() or (raw < 0).any() or not np.isclose(raw.sum(), 1.0, atol=1e-5):
                    raise ValueError("Invalid class-probability output")
                aligned = np.zeros(6)
                for label, probability in zip(classes, raw):
                    if label not in LABEL_TO_CATEGORY:
                        raise ValueError("Unknown probability class")
                    aligned[int(label)] = probability
                result["probabilities"] = aligned
            except Exception as exc:
                result["probability_note"] = f"Class probabilities unavailable: {exc}"
        else:
            result["probability_note"] = "This SVM was trained without probability estimation; no class probabilities are available."
        return result
    except (ArtifactError, InputValidationError):
        raise
    except Exception as exc:
        raise ArtifactError(f"Prediction failed: {exc}") from exc

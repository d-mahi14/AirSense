"""Prompt 13: real-artifact inference, exact schema and graceful error cases."""
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src import prediction_service as service
from src.feature_engineering import prepare_modeling_dataset, extract_X_y, get_feature_lists


@pytest.fixture(scope="module")
def evidence():
    evaluation, error = service.load_evaluation()
    assert not error
    training = service.training_rows(service.load_dataset())
    bounds = service.input_ranges(training)
    example = training.dropna(subset=service.POLLUTANT_COLS).iloc[100]
    return evaluation, training, bounds, example


@pytest.mark.parametrize("candidate", list(service.MODEL_NAMES) + ["tuned_" + key for key in service.MODEL_NAMES])
def test_actual_predictions_match_saved_pipeline_and_training_feature_builder(evidence, candidate):
    evaluation, _, bounds, example = evidence
    pipeline = service.load_model(candidate, evaluation)
    values = {f: example[f] for f in service.POLLUTANT_COLS}
    frame, notes = service.build_model_input(pipeline, values, example.Date, bounds)
    prepared = prepare_modeling_dataset(pd.DataFrame([example]).reset_index(drop=True))
    expected, _ = extract_X_y(prepared, get_feature_lists(prepared))
    pd.testing.assert_frame_equal(frame, expected, check_dtype=True)
    assert list(frame) == list(pipeline.feature_names_in_)
    result = service.predict(pipeline, frame)
    assert result["category"] == service.AQI_CATEGORIES[int(pipeline.predict(frame)[0])]
    if candidate.endswith("svm"):
        assert result["probabilities"] is None
        assert "without probability" in result["probability_note"]
    else:
        np.testing.assert_allclose(result["probabilities"], pipeline.predict_proba(frame)[0], atol=1e-7)
        assert np.isclose(result["probabilities"].sum(), 1.0)
    assert not notes["imputed_pollutants"]


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf"), "invalid", True, 1000000.0])
def test_invalid_pollutant_readings_rejected(evidence, invalid):
    evaluation, _, bounds, example = evidence
    pipeline = service.load_model("decision_tree", evaluation)
    readings = {f: example[f] for f in service.POLLUTANT_COLS}
    readings["PM2.5"] = invalid
    with pytest.raises(service.InputValidationError):
        service.build_model_input(pipeline, readings, example.Date, bounds)


def test_blank_measurements_are_imputed_by_original_pipeline(evidence):
    evaluation, _, bounds, example = evidence
    pipeline = service.load_model("random_forest", evaluation)
    readings = {f: example[f] for f in service.POLLUTANT_COLS}
    readings["PM10"] = None
    frame, notes = service.build_model_input(pipeline, readings, example.Date, bounds)
    assert pd.isna(frame.PM10.iloc[0])
    assert notes["imputed_pollutants"] == ["PM10"]
    assert service.predict(pipeline, frame)["category"] in service.AQI_CATEGORIES
    with pytest.raises(service.InputValidationError, match="at least one"):
        service.build_model_input(pipeline, {}, example.Date, bounds)


@pytest.mark.parametrize("invalid_date", ["2020-02-30", "2100-01-01", "2014-12-31", None])
def test_invalid_measurement_dates_rejected(evidence, invalid_date):
    evaluation, _, bounds, example = evidence
    pipeline = service.load_model("decision_tree", evaluation)
    with pytest.raises(service.InputValidationError, match="date"):
        service.build_model_input(pipeline, {f: example[f] for f in service.POLLUTANT_COLS}, invalid_date, bounds)


def test_extra_weather_input_rejected(evidence):
    evaluation, _, bounds, example = evidence
    pipeline = service.load_model("decision_tree", evaluation)
    readings = {f: example[f] for f in service.POLLUTANT_COLS}
    readings["temperature"] = 25
    with pytest.raises(service.InputValidationError, match="Unexpected"):
        service.build_model_input(pipeline, readings, example.Date, bounds)


def test_model_load_is_cached_and_does_not_refit(evidence, monkeypatch):
    evaluation, *_ = evidence
    service._load_model_cached.clear()
    original = joblib.load
    calls = []
    def tracked(path):
        calls.append(str(path))
        return original(path)
    monkeypatch.setattr(joblib, "load", tracked)
    a = service.load_model("decision_tree", evaluation)
    b = service.load_model("decision_tree", evaluation)
    assert a is b
    assert len(calls) == 1


def test_missing_corrupt_or_changed_models_report_errors(tmp_path, monkeypatch, evidence):
    evaluation, *_ = evidence
    monkeypatch.setattr(service, "MODEL_DIR", tmp_path)
    with pytest.raises(service.ArtifactError, match="missing"):
        service.load_model("decision_tree", evaluation)
    path = tmp_path / "decision_tree.joblib"
    path.write_bytes(b"not a model")
    with pytest.raises(service.ArtifactError, match="checksum"):
        service.load_model("decision_tree", evaluation)
    with pytest.raises(service.ArtifactError, match="Could not load"):
        service.load_model("decision_tree", {})


def test_stale_evaluation_metrics_are_not_displayed(tmp_path, monkeypatch, evidence):
    evaluation, *_ = evidence
    monkeypatch.setattr(service, "MODEL_DIR", tmp_path)
    (tmp_path / "selection_manifest.json").write_text("{}")
    values, error = service.load_evaluation()
    assert not values and "stale" in error


def test_class_probabilities_align_with_classifier_label_order():
    class Model:
        feature_names_in_ = np.array(["PM2.5"])
        named_steps = {"classifier": object()}
        classes_ = np.array([5, 0])
        def predict(self, X): return np.array([5])
        def predict_proba(self, X): return np.array([[0.8, 0.2]])
    result = service.predict(Model(), pd.DataFrame({"PM2.5": [1.0]}))
    assert result["category"] == "Severe"
    np.testing.assert_allclose(result["probabilities"], [0.2, 0, 0, 0, 0, 0.8])

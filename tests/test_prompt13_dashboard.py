"""End-to-end Streamlit interactions over saved prompt 11/12 artifacts."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from sklearn.pipeline import Pipeline
from src import prediction_service as service

ROOT = Path(__file__).resolve().parents[1]


def test_all_pages_render_with_actual_results_and_no_training(monkeypatch):
    def no_fit(*args, **kwargs):
        raise AssertionError("Dashboard must not train models")
    monkeypatch.setattr(Pipeline, "fit", no_fit)
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not app.exception and not app.error
    evaluation, error = service.load_evaluation()
    assert not error
    assert any(m.value == f"{evaluation['candidates'][evaluation['selected_candidate']]['test_metrics']['macro_f1']:.4f}" for m in app.metric)
    for page in ["Air Quality Prediction", "Model Comparison", "Pollution Analysis", "Research and Methodology"]:
        app.sidebar.radio(key="navigation").set_value(page).run(timeout=30)
        assert not app.exception and not app.error, page
    app.sidebar.radio(key="navigation").set_value("Model Comparison").run(timeout=30)
    for variant in ["Baseline", "Tuned"]:
        app.radio(key="comparison_variant").set_value(variant).run(timeout=30)
        actual = app.dataframe[0].value
        for _, row in actual.iterrows():
            candidate = service.candidate_key(row.Model, variant.lower())
            assert row["Macro F1"] == evaluation["candidates"][candidate]["test_metrics"]["macro_f1"]
        assert not app.exception and not app.error
    app.sidebar.radio(key="navigation").set_value("Pollution Analysis").run(timeout=30)
    app.radio(key="analysis_period").set_value("Full historical dataset").run(timeout=30)
    app.multiselect(key="correlation_fields").set_value([]).run(timeout=30)
    assert any("at least two" in info.value for info in app.info)
    assert not app.exception and not app.error


def test_prediction_form_all_eight_models_and_empty_input_error():
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    app.sidebar.radio(key="navigation").set_value("Air Quality Prediction").run(timeout=30)
    submit = next(b for b in app.button if b.label == "Classify sample")
    submit.click().run(timeout=30)
    assert any("at least one" in error.value for error in app.error)
    assert not app.exception
    for variant in ["Baseline", "Tuned"]:
        app.radio(key="prediction_variant").set_value(variant).run(timeout=30)
        for name in service.MODEL_NAMES.values():
            app.selectbox(key="prediction_model").set_value(name).run(timeout=30)
            assert len(app.number_input) == 12
            app.button(key="load_example").click().run(timeout=30)
            next(b for b in app.button if b.label == "Classify sample").click().run(timeout=30)
            assert not app.exception and not app.error, (variant, name)
            stored = app.session_state["last_prediction"]
            assert stored["candidate"] == service.candidate_key(name, variant.lower())
            expected = service.predict(service.load_model(stored["candidate"], service.load_evaluation()[0]), stored["frame"])
            assert stored["result"]["category"] == expected["category"]
            assert not stored["notes"]["imputed_pollutants"]
            if name == "SVM":
                assert stored["result"]["probabilities"] is None
                assert any("without probability" in info.value for info in app.info)
            else:
                np.testing.assert_allclose(stored["result"]["probabilities"], expected["probabilities"])


def test_missing_models_are_graceful_in_prediction_page(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "MODEL_DIR", tmp_path)
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    app.sidebar.radio(key="navigation").set_value("Air Quality Prediction").run(timeout=30)
    assert not app.exception
    assert any("missing" in error.value.lower() for error in app.error)
    assert not app.number_input


def test_missing_raw_data_does_not_break_metrics_or_navigation(monkeypatch):
    def unavailable():
        raise service.ArtifactError("Dataset unavailable for this test")
    monkeypatch.setattr(service, "load_dataset", unavailable)
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    app.sidebar.radio(key="navigation").set_value("Pollution Analysis").run(timeout=30)
    assert not app.exception
    assert any("Dataset unavailable" in warning.value for warning in app.warning)
    app.sidebar.radio(key="navigation").set_value("Model Comparison").run(timeout=30)
    assert not app.exception and not app.error
    assert len(app.dataframe[0].value) == 4


def test_missing_shap_does_not_break_other_analysis(tmp_path, monkeypatch):
    from shutil import copyfile
    reports = tmp_path / "reports"
    features = reports / "feature_analysis"
    features.mkdir(parents=True)
    copyfile(service.REPORTS_DIR / "evaluation_results.json", reports / "evaluation_results.json")
    for name in ["sampling_manifest.json", "random_forest_mdi.csv", "random_forest_permutation.csv", "xgboost_gain.csv", "xgboost_permutation.csv"]:
        copyfile(service.REPORTS_DIR / "feature_analysis" / name, features / name)
    monkeypatch.setattr(service, "REPORTS_DIR", reports)
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    app.sidebar.radio(key="navigation").set_value("Pollution Analysis").run(timeout=30)
    assert not app.exception and not app.error
    assert any("xgboost_shap.csv" in info.value for info in app.info)

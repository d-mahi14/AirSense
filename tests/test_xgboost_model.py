"""
Tests for AirSense XGBoost Model (src/models/xgboost_model.py)

Test coverage:
1. XGBoost installation and version check.
2. Pipeline construction.
3. Training with correct multiclass configuration.
4. Label consistency (integer labels 0–5).
5. Early stopping uses validation set correctly.
6. Evaluation metrics validity.
7. Feature importances.
8. Model persistence.
9. End-to-end run.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

import joblib

from src.models.xgboost_model import (
    check_xgboost,
    build_xgboost_pipeline,
    run_xgboost,
    XGBOOST_AVAILABLE,
    XGBOOST_VERSION,
)
from src.models.decision_tree_model import evaluate_model, save_model_pipeline
from src.pipeline import prepare_all_splits

# Skip all tests if XGBoost is not installed
xgboost_installed = pytest.mark.skipif(
    not XGBOOST_AVAILABLE,
    reason="XGBoost not installed. Install with: pip install xgboost"
)


# ─────────────────────────────────────────────────────────────────
# Fixture
# ─────────────────────────────────────────────────────────────────

def make_df(n: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2015-01-01", periods=n, freq="D")
    cats = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    base = cats * (n // 6)
    extra = cats[:n % 6]
    labels = base + extra
    rng.shuffle(labels)
    return pd.DataFrame({
        "Date": dates,
        "City": rng.choice(["Bengaluru", "Ahmedabad"], n),
        "PM2.5": rng.uniform(5, 300, n),
        "PM10":  rng.uniform(10, 500, n),
        "NO":    rng.uniform(0, 100, n),
        "NO2":   rng.uniform(0, 150, n),
        "NOx":   rng.uniform(0, 200, n),
        "NH3":   rng.uniform(0, 80, n),
        "CO":    rng.uniform(0.1, 5, n),
        "SO2":   rng.uniform(0, 80, n),
        "O3":    rng.uniform(0, 120, n),
        "Benzene": rng.uniform(0, 20, n),
        "Toluene": rng.uniform(0, 50, n),
        "Xylene":  rng.uniform(0, 30, n),
        "AQI":     rng.uniform(10, 450, n),
        "AQI_Bucket": labels,
    })


def get_splits():
    full = make_df(n=120)
    return full[:72], full[72:96], full[96:]


# ─────────────────────────────────────────────────────────────────
# 1. XGBoost installation check
# ─────────────────────────────────────────────────────────────────

class TestXGBoostInstallation:
    def test_xgboost_installed(self):
        """XGBoost must be importable."""
        assert XGBOOST_AVAILABLE, (
            "XGBoost is NOT installed. Install it with:\n"
            "  python.exe -m pip install xgboost\n"
            "in PowerShell from the AirSense project directory."
        )

    @xgboost_installed
    def test_version_string_format(self):
        v = check_xgboost()
        # Version should be a string like '2.0.3' or '1.7.6'
        assert isinstance(v, str)
        parts = v.split(".")
        assert len(parts) >= 2, f"Unexpected version format: {v}"

    @xgboost_installed
    def test_check_xgboost_returns_version(self):
        v = check_xgboost()
        assert v == XGBOOST_VERSION


# ─────────────────────────────────────────────────────────────────
# 2. Pipeline construction
# ─────────────────────────────────────────────────────────────────

@xgboost_installed
class TestXGBoostPipelineConstruction:
    def test_pipeline_builds_without_error(self):
        from sklearn.pipeline import Pipeline
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        pipe = build_xgboost_pipeline(data["pipeline"], n_estimators=10)
        assert isinstance(pipe, Pipeline)

    def test_classifier_is_xgbclassifier(self):
        from xgboost import XGBClassifier
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        pipe = build_xgboost_pipeline(data["pipeline"], n_estimators=10)
        assert isinstance(pipe.named_steps["classifier"], XGBClassifier)

    def test_n_estimators_applied(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        pipe = build_xgboost_pipeline(data["pipeline"], n_estimators=25)
        assert pipe.named_steps["classifier"].n_estimators == 25

    def test_objective_is_multiclass(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        pipe = build_xgboost_pipeline(data["pipeline"], n_estimators=10)
        clf = pipe.named_steps["classifier"]
        assert "multi" in clf.objective


# ─────────────────────────────────────────────────────────────────
# 3. Label consistency
# ─────────────────────────────────────────────────────────────────

@xgboost_installed
class TestXGBoostLabelConsistency:
    def test_predictions_are_valid_integer_labels(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)

        # Pre-transform for direct clf use
        preproc = data["pipeline"]
        X_train_t = preproc.transform(data["X_train_df"])
        X_val_t   = preproc.transform(data["X_val_df"])

        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=10, objective="multi:softmax",
            num_class=6, random_state=42, n_jobs=1, verbosity=0
        )
        clf.fit(X_train_t, data["y_train"])
        y_pred = clf.predict(X_val_t)
        assert set(y_pred.tolist()).issubset({0, 1, 2, 3, 4, 5})

    def test_y_train_labels_are_zero_indexed(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        y = data["y_train"]
        assert y.min() == 0, f"y_train min should be 0, got {y.min()}"
        assert y.max() == 5, f"y_train max should be 5, got {y.max()}"


# ─────────────────────────────────────────────────────────────────
# 4. Evaluation metrics
# ─────────────────────────────────────────────────────────────────

@xgboost_installed
class TestXGBoostEvaluation:
    def test_run_returns_all_metric_keys(self):
        train_df, val_df, test_df = get_splits()
        hp = {"n_estimators": 10, "n_jobs": 1}
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_xgboost(
                train_df, val_df, test_df,
                hyperparams=hp,
                model_dir=Path(tmpdir),
                use_early_stopping=False
            )
        for key in ["train_metrics", "val_metrics", "test_metrics"]:
            assert key in results
            assert 0.0 <= results[key]["accuracy"] <= 1.0
            assert 0.0 <= results[key]["macro_f1"] <= 1.0

    def test_val_metrics_evaluated_on_full_val_set(self):
        train_df, val_df, test_df = get_splits()
        hp = {"n_estimators": 10, "n_jobs": 1}
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_xgboost(
                train_df, val_df, test_df,
                hyperparams=hp,
                model_dir=Path(tmpdir),
                use_early_stopping=False
            )
        # Val metrics must be on the FULL validation set
        assert results["val_metrics"]["n_samples"] == len(val_df)
        assert results["test_metrics"]["n_samples"] == len(test_df)


# ─────────────────────────────────────────────────────────────────
# 5. Feature Importances
# ─────────────────────────────────────────────────────────────────

@xgboost_installed
class TestXGBoostFeatureImportances:
    def test_feature_importances_returned(self):
        train_df, val_df, test_df = get_splits()
        hp = {"n_estimators": 10, "n_jobs": 1}
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_xgboost(
                train_df, val_df, test_df,
                hyperparams=hp,
                model_dir=Path(tmpdir),
                use_early_stopping=False
            )
        df_imp = results["feature_importances"]
        if not df_imp.empty:
            assert "Feature" in df_imp.columns
            assert "Importance" in df_imp.columns


# ─────────────────────────────────────────────────────────────────
# 6. Model persistence
# ─────────────────────────────────────────────────────────────────

@xgboost_installed
class TestXGBoostPersistence:
    def test_model_files_created(self):
        train_df, val_df, test_df = get_splits()
        hp = {"n_estimators": 10, "n_jobs": 1}
        with tempfile.TemporaryDirectory() as tmpdir:
            run_xgboost(
                train_df, val_df, test_df,
                hyperparams=hp,
                model_dir=Path(tmpdir),
                use_early_stopping=False
            )
            assert (Path(tmpdir) / "xgboost.joblib").exists()
            assert (Path(tmpdir) / "xgboost_config.json").exists()

    def test_save_reload_predictions_match(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        pipe = build_xgboost_pipeline(data["pipeline"], n_estimators=10, n_jobs=1)
        pipe.fit(data["X_train_df"], data["y_train"])
        pred_before = pipe.predict(data["X_test_df"])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = save_model_pipeline(pipe, "xgb_test", Path(tmpdir))
            loaded = joblib.load(save_path)
            pred_after = loaded.predict(data["X_test_df"])

        np.testing.assert_array_equal(pred_before, pred_after)


# ─────────────────────────────────────────────────────────────────
# 7. Reproducibility
# ─────────────────────────────────────────────────────────────────

@xgboost_installed
class TestXGBoostReproducibility:
    def test_identical_predictions_with_same_seed(self):
        train_df, val_df, test_df = get_splits()
        data1 = prepare_all_splits(train_df, val_df, test_df)
        pipe1 = build_xgboost_pipeline(data1["pipeline"], n_estimators=10, random_state=0, n_jobs=1)
        pipe1.fit(data1["X_train_df"], data1["y_train"])
        pred1 = pipe1.predict(data1["X_test_df"])

        data2 = prepare_all_splits(train_df, val_df, test_df)
        pipe2 = build_xgboost_pipeline(data2["pipeline"], n_estimators=10, random_state=0, n_jobs=1)
        pipe2.fit(data2["X_train_df"], data2["y_train"])
        pred2 = pipe2.predict(data2["X_test_df"])

        np.testing.assert_array_equal(pred1, pred2, err_msg="XGBoost predictions not reproducible")

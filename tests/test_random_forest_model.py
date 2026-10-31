"""
Tests for AirSense Random Forest Model (src/models/random_forest_model.py)

Test coverage:
1. Pipeline construction with RandomForestClassifier.
2. Training completes with OOB score.
3. Feature importance extraction.
4. Predictions are valid labels.
5. Reproducibility.
6. Model persistence.
7. End-to-end run.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.models.random_forest_model import (
    build_random_forest_pipeline,
    get_feature_importances,
    run_random_forest,
)
from src.models.decision_tree_model import evaluate_model, save_model_pipeline
from src.pipeline import prepare_all_splits
from src.feature_engineering import prepare_modeling_dataset, get_feature_lists


# ─────────────────────────────────────────────────────────────────
# Fixture
# ─────────────────────────────────────────────────────────────────

def make_df(n: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2017-01-01", periods=n, freq="D")
    cats = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    base = cats * (n // 6)
    extra = cats[:n % 6]
    labels = base + extra
    rng.shuffle(labels)
    return pd.DataFrame({
        "Date": dates,
        "City": rng.choice(["Delhi", "Pune"], n),
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
# 1. Pipeline construction
# ─────────────────────────────────────────────────────────────────

class TestRFPipelineConstruction:
    def test_pipeline_is_sklearn_pipeline(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        pipe = build_random_forest_pipeline(data["pipeline"])
        assert isinstance(pipe, Pipeline)

    def test_classifier_is_random_forest(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        pipe = build_random_forest_pipeline(data["pipeline"])
        assert isinstance(pipe.named_steps["classifier"], RandomForestClassifier)

    def test_n_estimators_applied(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        pipe = build_random_forest_pipeline(data["pipeline"], n_estimators=50)
        assert pipe.named_steps["classifier"].n_estimators == 50

    def test_class_weight_balanced(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        pipe = build_random_forest_pipeline(data["pipeline"], class_weight="balanced")
        assert pipe.named_steps["classifier"].class_weight == "balanced"


# ─────────────────────────────────────────────────────────────────
# 2. Training and OOB Score
# ─────────────────────────────────────────────────────────────────

class TestRFTraining:
    def test_training_produces_oob_score(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        pipe = build_random_forest_pipeline(data["pipeline"], n_estimators=20)
        pipe.fit(data["X_train_df"], data["y_train"])
        rf = pipe.named_steps["classifier"]
        assert hasattr(rf, "oob_score_")
        assert 0.0 <= rf.oob_score_ <= 1.0

    def test_predictions_valid_class_labels(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        pipe = build_random_forest_pipeline(data["pipeline"], n_estimators=20)
        pipe.fit(data["X_train_df"], data["y_train"])
        y_pred = pipe.predict(data["X_val_df"])
        assert set(y_pred).issubset({0, 1, 2, 3, 4, 5})

    def test_reproducibility_same_predictions(self):
        train_df, val_df, test_df = get_splits()
        data1 = prepare_all_splits(train_df, val_df, test_df)
        pipe1 = build_random_forest_pipeline(data1["pipeline"], n_estimators=20, random_state=7)
        pipe1.fit(data1["X_train_df"], data1["y_train"])
        pred1 = pipe1.predict(data1["X_test_df"])

        data2 = prepare_all_splits(train_df, val_df, test_df)
        pipe2 = build_random_forest_pipeline(data2["pipeline"], n_estimators=20, random_state=7)
        pipe2.fit(data2["X_train_df"], data2["y_train"])
        pred2 = pipe2.predict(data2["X_test_df"])

        np.testing.assert_array_equal(pred1, pred2, err_msg="RF predictions not reproducible")


# ─────────────────────────────────────────────────────────────────
# 3. Feature Importances
# ─────────────────────────────────────────────────────────────────

class TestRFFeatureImportances:
    def setup_method(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        self.pipe = build_random_forest_pipeline(data["pipeline"], n_estimators=20)
        self.pipe.fit(data["X_train_df"], data["y_train"])
        self.feature_names = data["feature_names_out"]

    def test_feature_importances_returns_dataframe(self):
        df_imp = get_feature_importances(self.pipe, self.feature_names)
        assert isinstance(df_imp, pd.DataFrame)

    def test_importances_sum_to_approx_one(self):
        df_imp = get_feature_importances(self.pipe, self.feature_names)
        total = df_imp["Importance"].sum()
        assert abs(total - 1.0) < 0.01, f"Feature importances sum to {total}, expected ~1.0"

    def test_importances_non_negative(self):
        df_imp = get_feature_importances(self.pipe, self.feature_names)
        assert (df_imp["Importance"] >= 0).all()

    def test_importances_sorted_descending(self):
        df_imp = get_feature_importances(self.pipe, self.feature_names)
        assert df_imp["Importance"].is_monotonic_decreasing


# ─────────────────────────────────────────────────────────────────
# 4. Model Persistence
# ─────────────────────────────────────────────────────────────────

class TestRFPersistence:
    def test_save_and_reload_predictions_match(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        pipe = build_random_forest_pipeline(data["pipeline"], n_estimators=10)
        pipe.fit(data["X_train_df"], data["y_train"])
        pred_before = pipe.predict(data["X_test_df"])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = save_model_pipeline(pipe, "rf_test", Path(tmpdir))
            loaded = joblib.load(save_path)
            pred_after = loaded.predict(data["X_test_df"])

        np.testing.assert_array_equal(pred_before, pred_after)


# ─────────────────────────────────────────────────────────────────
# 5. End-to-end run_random_forest
# ─────────────────────────────────────────────────────────────────

class TestRunRandomForest:
    def test_run_returns_expected_keys(self):
        train_df, val_df, test_df = get_splits()
        hp = {"n_estimators": 15, "n_jobs": 1}
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_random_forest(
                train_df, val_df, test_df,
                hyperparams=hp,
                model_dir=Path(tmpdir)
            )
        for key in ["pipeline", "train_metrics", "val_metrics", "test_metrics",
                    "oob_score", "feature_importances", "config"]:
            assert key in results

    def test_model_files_created(self):
        train_df, val_df, test_df = get_splits()
        hp = {"n_estimators": 10, "n_jobs": 1}
        with tempfile.TemporaryDirectory() as tmpdir:
            run_random_forest(train_df, val_df, test_df, hyperparams=hp, model_dir=Path(tmpdir))
            assert (Path(tmpdir) / "random_forest.joblib").exists()
            assert (Path(tmpdir) / "random_forest_config.json").exists()

    def test_oob_score_is_valid(self):
        train_df, val_df, test_df = get_splits()
        hp = {"n_estimators": 10, "n_jobs": 1}
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_random_forest(
                train_df, val_df, test_df, hyperparams=hp, model_dir=Path(tmpdir)
            )
        oob = results["oob_score"]
        if oob is not None:
            assert 0.0 <= oob <= 1.0

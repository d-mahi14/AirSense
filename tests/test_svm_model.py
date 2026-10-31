"""
Tests for AirSense SVM Model (src/models/svm_model.py)

Test coverage:
1. Pipeline construction with SVC.
2. Training and prediction on scaled features.
3. Unseen test set evaluation (full val/test even with subsampled training).
4. Subsampling disclosure.
5. Model persistence.
6. End-to-end run.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

import joblib
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from src.models.svm_model import build_svm_pipeline, run_svm
from src.models.decision_tree_model import evaluate_model, save_model_pipeline
from src.pipeline import prepare_all_splits


# ─────────────────────────────────────────────────────────────────
# Fixture
# ─────────────────────────────────────────────────────────────────

def make_df(n: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2016-01-01", periods=n, freq="D")
    cats = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    base = cats * (n // 6)
    extra = cats[:n % 6]
    labels = base + extra
    rng.shuffle(labels)
    return pd.DataFrame({
        "Date": dates,
        "City": rng.choice(["Delhi", "Hyderabad"], n),
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

class TestSVMPipelineConstruction:
    def test_pipeline_is_sklearn_pipeline(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        pipe = build_svm_pipeline(data["pipeline"])
        assert isinstance(pipe, Pipeline)

    def test_classifier_is_svc(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        pipe = build_svm_pipeline(data["pipeline"])
        assert isinstance(pipe.named_steps["classifier"], SVC)

    def test_kernel_parameter_applied(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        pipe = build_svm_pipeline(data["pipeline"], kernel="linear")
        assert pipe.named_steps["classifier"].kernel == "linear"

    def test_c_parameter_applied(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        pipe = build_svm_pipeline(data["pipeline"], C=0.5)
        assert pipe.named_steps["classifier"].C == 0.5

    def test_class_weight_balanced(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        pipe = build_svm_pipeline(data["pipeline"], class_weight="balanced")
        assert pipe.named_steps["classifier"].class_weight == "balanced"


# ─────────────────────────────────────────────────────────────────
# 2. Training (linear kernel for speed in tests)
# ─────────────────────────────────────────────────────────────────

class TestSVMTraining:
    def test_linear_svm_trains_and_predicts(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        pipe = build_svm_pipeline(data["pipeline"], kernel="linear", C=1.0)
        pipe.fit(data["X_train_df"], data["y_train"])
        y_pred = pipe.predict(data["X_val_df"])
        assert set(y_pred).issubset({0, 1, 2, 3, 4, 5})

    def test_predictions_are_integers(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        pipe = build_svm_pipeline(data["pipeline"], kernel="linear", C=1.0)
        pipe.fit(data["X_train_df"], data["y_train"])
        y_pred = pipe.predict(data["X_val_df"])
        for v in y_pred:
            assert isinstance(v, (int, np.integer)), f"Prediction {v} is not integer"


# ─────────────────────────────────────────────────────────────────
# 3. Subsampling disclosure
# ─────────────────────────────────────────────────────────────────

class TestSVMSubsampling:
    def test_subsampling_reduces_training_size(self):
        """Subsampled run trains on fewer samples but evaluates on full val/test."""
        train_df, val_df, test_df = get_splits()
        hp = {"kernel": "linear", "C": 1.0}
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_svm(
                train_df, val_df, test_df,
                hyperparams=hp,
                model_dir=Path(tmpdir),
                max_train_samples=30
            )
        assert results["subsampled"] is True
        # val/test are never subsampled
        assert results["val_metrics"]["n_samples"] == len(val_df)
        assert results["test_metrics"]["n_samples"] == len(test_df)


# ─────────────────────────────────────────────────────────────────
# 4. Evaluation metrics
# ─────────────────────────────────────────────────────────────────

class TestSVMEvaluation:
    def test_accuracy_in_valid_range(self):
        train_df, val_df, test_df = get_splits()
        hp = {"kernel": "linear", "C": 1.0}
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_svm(train_df, val_df, test_df, hyperparams=hp, model_dir=Path(tmpdir))
        assert 0.0 <= results["val_metrics"]["accuracy"] <= 1.0
        assert 0.0 <= results["test_metrics"]["accuracy"] <= 1.0

    def test_macro_f1_in_valid_range(self):
        train_df, val_df, test_df = get_splits()
        hp = {"kernel": "linear", "C": 1.0}
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_svm(train_df, val_df, test_df, hyperparams=hp, model_dir=Path(tmpdir))
        assert 0.0 <= results["val_metrics"]["macro_f1"] <= 1.0


# ─────────────────────────────────────────────────────────────────
# 5. Model persistence
# ─────────────────────────────────────────────────────────────────

class TestSVMPersistence:
    def test_model_files_created(self):
        train_df, val_df, test_df = get_splits()
        hp = {"kernel": "linear", "C": 0.5}
        with tempfile.TemporaryDirectory() as tmpdir:
            run_svm(train_df, val_df, test_df, hyperparams=hp, model_dir=Path(tmpdir))
            assert (Path(tmpdir) / "svm.joblib").exists()
            assert (Path(tmpdir) / "svm_config.json").exists()

    def test_save_and_reload_predictions_match(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        pipe = build_svm_pipeline(data["pipeline"], kernel="linear", C=1.0)
        pipe.fit(data["X_train_df"], data["y_train"])
        pred_before = pipe.predict(data["X_test_df"])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = save_model_pipeline(pipe, "svm_test", Path(tmpdir))
            loaded = joblib.load(save_path)
            pred_after = loaded.predict(data["X_test_df"])

        np.testing.assert_array_equal(pred_before, pred_after)

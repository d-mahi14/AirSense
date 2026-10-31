"""
Tests for AirSense Decision Tree Model (src/models/decision_tree_model.py)

Test coverage:
1. Pipeline builds correctly with DecisionTreeClassifier.
2. Model trains on synthetic data without errors.
3. Predictions are valid class labels.
4. Evaluation metrics are within expected bounds.
5. Class weight parameter is applied.
6. Model persistence (save/load).
7. Reproducibility via random_state.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from src.models.decision_tree_model import (
    build_decision_tree_pipeline,
    train_decision_tree,
    evaluate_model,
    save_model_pipeline,
    run_decision_tree,
    print_evaluation_report,
)
from src.pipeline import build_preprocessing_pipeline, prepare_all_splits
from src.feature_engineering import prepare_modeling_dataset, get_feature_lists, extract_X_y


# ─────────────────────────────────────────────────────────────────
# Fixture: Synthetic data
# ─────────────────────────────────────────────────────────────────

def make_df(n: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    cats  = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    # Ensure all 6 classes appear at least once
    base  = cats * (n // 6)
    extra = cats[:n % 6]
    labels = base + extra
    rng.shuffle(labels)
    return pd.DataFrame({
        "Date": dates,
        "City": rng.choice(["Delhi", "Mumbai"], n),
        "PM2.5": rng.uniform(5, 300, n),
        "PM10":  rng.uniform(10, 500, n),
        "NO":    rng.uniform(0, 100, n),
        "NO2":   rng.uniform(0, 150, n),
        "NOx":   rng.uniform(0, 200, n),
        "NH3":   rng.uniform(0, 80, n),
        "CO":    rng.uniform(0, 5, n),
        "SO2":   rng.uniform(0, 80, n),
        "O3":    rng.uniform(0, 120, n),
        "Benzene": rng.uniform(0, 20, n),
        "Toluene": rng.uniform(0, 50, n),
        "Xylene":  rng.uniform(0, 30, n),
        "AQI":     rng.uniform(10, 450, n),
        "AQI_Bucket": labels,
    })


def get_splits(seed=42):
    full = make_df(n=120, seed=seed)
    return full[:72], full[72:96], full[96:]


# ─────────────────────────────────────────────────────────────────
# 1. Pipeline construction
# ─────────────────────────────────────────────────────────────────

class TestDecisionTreePipelineConstruction:
    def test_pipeline_returns_sklearn_pipeline(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        preproc = data["pipeline"]
        full_pipe = build_decision_tree_pipeline(preproc)
        assert isinstance(full_pipe, Pipeline)

    def test_classifier_is_decision_tree(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        preproc = data["pipeline"]
        full_pipe = build_decision_tree_pipeline(preproc)
        assert isinstance(full_pipe.named_steps["classifier"], DecisionTreeClassifier)

    def test_hyperparams_applied_to_classifier(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        preproc = data["pipeline"]
        full_pipe = build_decision_tree_pipeline(
            preproc, max_depth=5, criterion="entropy", class_weight=None
        )
        clf = full_pipe.named_steps["classifier"]
        assert clf.max_depth == 5
        assert clf.criterion == "entropy"
        assert clf.class_weight is None


# ─────────────────────────────────────────────────────────────────
# 2. Training
# ─────────────────────────────────────────────────────────────────

class TestDecisionTreeTraining:
    def test_training_completes_without_error(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        preproc = data["pipeline"]
        full_pipe = build_decision_tree_pipeline(preproc)
        X_train = data["X_train_df"]
        y_train = data["y_train"]
        fitted, t = train_decision_tree(full_pipe, X_train, y_train)
        assert fitted is not None
        assert t >= 0.0

    def test_predictions_are_valid_class_labels(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        preproc = data["pipeline"]
        full_pipe = build_decision_tree_pipeline(preproc)
        fitted, _ = train_decision_tree(full_pipe, data["X_train_df"], data["y_train"])
        y_pred = fitted.predict(data["X_val_df"])
        assert set(y_pred).issubset({0, 1, 2, 3, 4, 5})

    def test_training_is_reproducible(self):
        train_df, val_df, test_df = get_splits()

        data1 = prepare_all_splits(train_df, val_df, test_df)
        pipe1 = build_decision_tree_pipeline(data1["pipeline"], random_state=42)
        pipe1.fit(data1["X_train_df"], data1["y_train"])
        pred1 = pipe1.predict(data1["X_test_df"])

        data2 = prepare_all_splits(train_df, val_df, test_df)
        pipe2 = build_decision_tree_pipeline(data2["pipeline"], random_state=42)
        pipe2.fit(data2["X_train_df"], data2["y_train"])
        pred2 = pipe2.predict(data2["X_test_df"])

        np.testing.assert_array_equal(pred1, pred2, err_msg="DT predictions not reproducible")


# ─────────────────────────────────────────────────────────────────
# 3. Evaluation metrics
# ─────────────────────────────────────────────────────────────────

class TestDecisionTreeEvaluation:
    def setup_method(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        preproc = data["pipeline"]
        pipe = build_decision_tree_pipeline(preproc)
        pipe.fit(data["X_train_df"], data["y_train"])
        self.pipe = pipe
        self.data = data

    def test_metrics_dict_has_required_keys(self):
        metrics = evaluate_model(self.pipe, self.data["X_val_df"], self.data["y_val"], "Val")
        required_keys = [
            "accuracy", "macro_precision", "macro_recall", "macro_f1",
            "weighted_f1", "per_class_report", "confusion_matrix", "class_labels"
        ]
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_accuracy_is_valid_fraction(self):
        metrics = evaluate_model(self.pipe, self.data["X_val_df"], self.data["y_val"], "Val")
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_macro_f1_is_valid_fraction(self):
        metrics = evaluate_model(self.pipe, self.data["X_val_df"], self.data["y_val"], "Val")
        assert 0.0 <= metrics["macro_f1"] <= 1.0

    def test_confusion_matrix_is_square(self):
        metrics = evaluate_model(self.pipe, self.data["X_val_df"], self.data["y_val"], "Val")
        cm = np.array(metrics["confusion_matrix"])
        assert cm.ndim == 2
        assert cm.shape[0] == cm.shape[1]

    def test_print_evaluation_runs_without_error(self, capsys):
        metrics = evaluate_model(self.pipe, self.data["X_val_df"], self.data["y_val"], "Val")
        print_evaluation_report(metrics)
        captured = capsys.readouterr()
        assert "Accuracy" in captured.out


# ─────────────────────────────────────────────────────────────────
# 4. Model persistence
# ─────────────────────────────────────────────────────────────────

class TestDecisionTreePersistence:
    def test_save_and_reload_pipeline(self):
        train_df, val_df, test_df = get_splits()
        data = prepare_all_splits(train_df, val_df, test_df)
        pipe = build_decision_tree_pipeline(data["pipeline"])
        pipe.fit(data["X_train_df"], data["y_train"])
        pred_before = pipe.predict(data["X_test_df"])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = save_model_pipeline(pipe, "dt_test", Path(tmpdir))
            assert save_path.exists()

            loaded = joblib.load(save_path)
            pred_after = loaded.predict(data["X_test_df"])

        np.testing.assert_array_equal(pred_before, pred_after,
                                      err_msg="Predictions differ after save/reload")


# ─────────────────────────────────────────────────────────────────
# 5. End-to-end run_decision_tree
# ─────────────────────────────────────────────────────────────────

class TestRunDecisionTree:
    def test_run_returns_expected_keys(self):
        train_df, val_df, test_df = get_splits()
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_decision_tree(train_df, val_df, test_df, model_dir=Path(tmpdir))
        for key in ["pipeline", "train_metrics", "val_metrics", "test_metrics", "config"]:
            assert key in results

    def test_model_files_saved(self):
        train_df, val_df, test_df = get_splits()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_decision_tree(train_df, val_df, test_df, model_dir=Path(tmpdir))
            assert (Path(tmpdir) / "decision_tree.joblib").exists()
            assert (Path(tmpdir) / "decision_tree_config.json").exists()

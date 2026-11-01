"""Regression tests for prompt 11/12 leakage and provenance safeguards."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src import experiment as exp
from src.evaluation import compute_metrics, select_best_model
from src.tuning import date_blocked_cv, run_search
from src.explainability import get_tree_importance, get_permutation_importance


def test_metrics_keep_six_classes_when_test_class_is_absent():
    result = compute_metrics([0, 0, 1], [0, 1, 1])
    expected = f1_score([0, 0, 1], [0, 1, 1], labels=list(range(6)), average="macro", zero_division=0)
    assert result["macro_f1"] == round(expected, 6)
    assert np.asarray(result["confusion_matrix"]).shape == (6, 6)
    assert sum(map(sum, result["confusion_matrix"])) == 3


@pytest.mark.parametrize("truth,pred", [([], []), ([0], [6]), ([0, 1], [0])])
def test_invalid_metrics_are_rejected(truth, pred):
    with pytest.raises(ValueError):
        compute_metrics(truth, pred)


def test_temporal_folds_keep_whole_dates_and_gap():
    dates = pd.Series(np.repeat(pd.date_range("2018-01-01", periods=100), 3))
    folds = date_blocked_cv(dates, n_splits=3, gap_days=7)
    for train, val in folds:
        assert dates.iloc[train].max() < dates.iloc[val].min() - pd.Timedelta(days=7)
        assert not set(dates.iloc[train]) & set(dates.iloc[val])
        for day in dates.iloc[val].unique():
            assert set(np.flatnonzero((dates == day).to_numpy())).issubset(set(val))
    assert set(folds[0][0]).issubset(set(folds[1][0]))


def test_selection_uses_validation_even_when_test_ranking_reverses():
    candidates = {
        "A": {"validation_metrics": {"macro_f1": 0.8}, "test_metrics": {"macro_f1": 0.1}},
        "B": {"validation_metrics": {"macro_f1": 0.7}, "test_metrics": {"macro_f1": 0.99}},
        "failed": {"error": "unavailable"},
    }
    assert exp.choose_validation(candidates) == "A"
    with pytest.raises(ValueError, match="validation"):
        select_best_model(pd.DataFrame(), split="Test")


class FitRecorder(BaseEstimator, TransformerMixin):
    seen = []

    def fit(self, X, y=None):
        type(self).seen.append(set(X.index))
        return self

    def transform(self, X):
        return X


def test_preprocessor_fits_only_fold_training_rows():
    X = pd.DataFrame({"signal": np.arange(120)})
    y = pd.Series(np.arange(120) % 6)
    dates = pd.date_range("2018-01-01", periods=120)
    FitRecorder.seen = []
    pipe = Pipeline([("pre", FitRecorder()), ("classifier", DummyClassifier(strategy="most_frequent"))])
    _, cv, _ = run_search(pipe, {"classifier__strategy": ["most_frequent"]}, X, y, dates, n_iter=1, n_jobs=1)
    assert len(cv) == 1
    folds = date_blocked_cv(dates)
    assert FitRecorder.seen == [set(tr) for tr, _ in folds]
    for seen, (_, val) in zip(FitRecorder.seen, folds):
        assert not seen & set(val)


class RecordingDummy(DummyClassifier):
    def fit(self, X, y, sample_weight=None):
        self.n_fit_rows_ = len(X)
        return super().fit(X, y, sample_weight=sample_weight)


@pytest.fixture
def fake_experiment(monkeypatch, tmp_path):
    models, reports = tmp_path / "models", tmp_path / "reports"
    monkeypatch.setattr(exp, "MODELS", models)
    monkeypatch.setattr(exp, "REPORTS", reports)
    monkeypatch.setattr(exp, "identity", lambda: {"protocol": "test", "dataset_sha256": "data"})
    requested = []

    def data(split):
        requested.append(split)
        if split == "test":
            raise AssertionError("Test split must never be requested")
        X = pd.DataFrame({"x": np.arange(120)})
        y = pd.Series(np.arange(120) % 6)
        df = pd.DataFrame({"Date": pd.date_range("2018-01-01", periods=120)})
        return df, X, y, {"numerical": ["x"], "categorical": [], "all": ["x"]}

    monkeypatch.setattr(exp, "data_for", data)
    monkeypatch.setattr(exp, "make_pipeline", lambda *args: Pipeline([("classifier", RecordingDummy())]))
    return requested, models, reports


def test_training_never_evaluates_test_and_records_failure(fake_experiment, monkeypatch):
    requests, models, _ = fake_experiment
    def builder(key, _):
        if key == "svm":
            raise RuntimeError("Deliberate model failure")
        return Pipeline([("classifier", DummyClassifier())])
    monkeypatch.setattr(exp, "make_pipeline", builder)
    result = exp.fit_baselines()
    assert requests == ["train", "validation"]
    assert "error" in result["svm"]
    assert "test_metrics" not in result["decision_tree"]
    selection = exp.read(models / "selection_manifest.json")
    assert "svm" not in selection["candidates"]
    assert not (models / "svm.joblib").exists()


def test_tampered_model_is_rejected_before_test_access(fake_experiment):
    requests, models, _ = fake_experiment
    exp.fit_baselines()
    with (models / "decision_tree.joblib").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="changed since selection"):
        exp.evaluate_all()
    assert "test" not in requests


def test_tuning_does_not_read_test_and_refits_all_training_rows(fake_experiment, monkeypatch):
    import src.tuning as tuning
    requests, models, reports = fake_experiment
    reports.mkdir()
    exp.fit_baselines()
    monkeypatch.setattr(tuning, "get_search_spaces", lambda: {k: {"classifier__strategy": ["most_frequent"]} for k in exp.NAMES})
    monkeypatch.setattr(tuning, "run_search", lambda *a, **kw: ({"classifier__strategy": "most_frequent"}, pd.DataFrame({"mean_test_score": [0.1]}), 0.01))
    results = exp.tune_all()
    assert "test" not in requests
    assert all(r["used_n_train"] == 120 for r in results.values())
    assert all("test_metrics" not in r for r in results.values())
    assert all((reports / "tuning" / f"{k}_cv_results.csv").exists() for k in exp.NAMES)
    import joblib
    assert all(joblib.load(models / f"{name}.joblib").named_steps["classifier"].n_fit_rows_ == 120 for name in results)


def test_permutation_and_tree_importance_follow_real_signal():
    rng = np.random.RandomState(42)
    X = pd.DataFrame({"signal": np.tile(np.arange(6), 100), "noise": rng.normal(size=600)})
    y = X.signal.astype(int)
    pipe = Pipeline([("classifier", DecisionTreeClassifier(max_depth=6, random_state=42))]).fit(X.iloc[:300], y.iloc[:300])
    importance = get_tree_importance(pipe, X.columns.tolist())
    assert importance.feature.iloc[0] == "signal"
    perm = get_permutation_importance(pipe, X.iloc[300:], y.iloc[300:], X.columns.tolist(), n_repeats=2, n_jobs=1)
    assert perm.feature.iloc[0] == "signal"
    assert perm.importance_mean.iloc[0] > 0.5
    with pytest.raises(ValueError, match="dimensions"):
        get_tree_importance(pipe, ["wrong"])


def test_final_evaluation_uses_same_rows_and_preserves_frozen_selection(fake_experiment, monkeypatch):
    requests, models, reports = fake_experiment
    exp.fit_baselines()
    before = (models / "selection_manifest.json").read_bytes()
    original = exp.data_for
    def final_data(split):
        if split != "test":
            return original(split)
        X = pd.DataFrame({"x": np.arange(24)})
        y = pd.Series(np.arange(24) % 6)
        df = pd.DataFrame({"City": ["A"] * 24, "Date": pd.date_range("2020-01-01", periods=24)})
        return df, X, y, {}
    monkeypatch.setattr(exp, "data_for", final_data)
    results = exp.evaluate_all()
    assert all("test_metrics" in v for v in results.values())
    assert all(v["test_metrics"]["n_samples"] == 24 for v in results.values())
    assert before == (models / "selection_manifest.json").read_bytes()
    predictions = pd.read_csv(reports / "test_predictions.csv")
    assert len(predictions) == 24 * 4
    row_sets = [set(zip(group.City, group.Date)) for _, group in predictions.groupby("candidate")]
    assert all(rows == row_sets[0] for rows in row_sets)
    assert len(list((reports / "figures" / "model_comparison").glob("confusion_matrix_*.png"))) == 4
    assert (reports / "model_comparison.md").exists()


def test_shap_requires_training_background_instead_of_using_evaluation_rows():
    from src.explainability import get_shap_values
    with pytest.raises(ValueError, match="training-only"):
        get_shap_values(None, np.zeros((4, 2)), ["a", "b"])

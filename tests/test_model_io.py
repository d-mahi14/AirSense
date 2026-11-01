"""Regression coverage for the Windows XGBoost snapshot loading failure."""
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest

from src import model_io
from src.experiment import ROOT, data_for


@pytest.mark.parametrize("name", ["xgboost", "tuned_xgboost"])
def test_native_bundle_bypasses_original_snapshot(monkeypatch, name):
    path = ROOT / "models" / f"{name}.joblib"
    original_load = joblib.load
    # Simulate the Windows failure: loading the original snapshot is forbidden.
    def guarded_load(filename, *args, **kwargs):
        if Path(filename) == path:
            raise AssertionError("Original XGBoost memory snapshot must not be loaded")
        return original_load(filename, *args, **kwargs)
    monkeypatch.setattr(model_io.joblib, "load", guarded_load)
    portable = model_io.load_pipeline(path)
    _, X, _, _ = data_for("test")
    saved = pd.read_csv(ROOT / "reports" / "test_predictions.csv")
    np.testing.assert_array_equal(portable.predict(X), saved.loc[saved.candidate == name, "y_pred"])
    np.testing.assert_allclose(portable.predict_proba(X).sum(axis=1), 1, atol=1e-6)


@pytest.mark.parametrize("suffix", [".trees.json", ".portable.joblib"])
def test_bundle_component_corruption_rejected(tmp_path, suffix):
    source = ROOT / "models" / "xgboost.joblib"
    for ext in (".joblib", ".trees.json", ".portable.joblib", ".portable.json"):
        shutil.copy2(source.with_suffix(ext), tmp_path / source.with_suffix(ext).name)
    changed = (tmp_path / source.name).with_suffix(suffix)
    changed.write_bytes(changed.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        model_io.load_pipeline(tmp_path / source.name)

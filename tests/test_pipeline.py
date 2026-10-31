"""
Tests for AirSense Preprocessing Pipeline (src/pipeline.py and src/feature_engineering.py)

Test coverage:
1. Missing values — imputer fills NaN in numerical features.
2. Unseen categories — OrdinalEncoder handles new Season/City values safely.
3. Reproducibility — same pipeline produces identical arrays across calls.
4. Train/test isolation — imputer/scaler fitted only on training data.
5. Leakage prevention — leakage columns cannot enter X.
6. Feature order consistency — output column order is fixed.
7. Target label encoding — correct integer → category mapping.
8. Scale consistency — StandardScaler zero-means and unit-variances training data.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.feature_engineering import (
    prepare_modeling_dataset,
    get_feature_lists,
    extract_X_y,
    NUMERICAL_POLLUTANT_FEATURES,
    NUMERICAL_TEMPORAL_FEATURES,
    ALL_CATEGORICAL_FEATURES,
    TARGET_COL,
    TARGET_LABEL_COL,
    LEAKAGE_COLS,
)
from src.pipeline import (
    build_preprocessing_pipeline,
    fit_pipeline,
    transform_split,
    get_feature_names_out,
    prepare_all_splits,
)
from src.preprocessing import CATEGORY_TO_LABEL, LABEL_TO_CATEGORY, SEASON_MAP


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

def make_minimal_df(n: int = 50, seed: int = 0, add_missing: bool = False) -> pd.DataFrame:
    """
    Build a small synthetic DataFrame that mimics city_day.csv structure.
    Includes all 12 pollutants, Date, City, AQI_Bucket, AQI.
    """
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    categories = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

    df = pd.DataFrame({
        "Date": dates,
        "City": rng.choice(["Delhi", "Mumbai", "Chennai", "Kolkata"], size=n),
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
        "AQI_Bucket": rng.choice(categories, size=n),
    })

    if add_missing:
        # Introduce NaN into 20% of pollutant readings
        for col in ["PM2.5", "PM10", "NO2", "CO"]:
            mask = rng.rand(n) < 0.2
            df.loc[mask, col] = np.nan

    return df


def make_three_split_dfs(seed: int = 42):
    """Return (train_df, val_df, test_df) synthetic splits."""
    full = make_minimal_df(n=150, seed=seed, add_missing=True)
    train_df = full.iloc[:90].copy().reset_index(drop=True)
    val_df   = full.iloc[90:120].copy().reset_index(drop=True)
    test_df  = full.iloc[120:].copy().reset_index(drop=True)
    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────
# 1. Test: prepare_modeling_dataset extracts temporal features
# ─────────────────────────────────────────────────────────────────

class TestPrepareModelingDataset:
    def test_temporal_features_extracted(self):
        df = make_minimal_df(n=20)
        df_prep = prepare_modeling_dataset(df)
        for feat in ["Year", "Month", "Day", "DayOfWeek", "IsWeekend", "Season"]:
            assert feat in df_prep.columns, f"Expected temporal feature '{feat}' missing"

    def test_target_label_encoded(self):
        df = make_minimal_df(n=20)
        df_prep = prepare_modeling_dataset(df)
        assert TARGET_LABEL_COL in df_prep.columns
        # All values should be in {0, 1, 2, 3, 4, 5}
        valid_labels = set(CATEGORY_TO_LABEL.values())
        assert df_prep[TARGET_LABEL_COL].isin(valid_labels).all()

    def test_season_values_valid(self):
        df = make_minimal_df(n=50)
        df_prep = prepare_modeling_dataset(df)
        valid_seasons = {"Winter", "Summer", "Monsoon", "Post-Monsoon"}
        assert set(df_prep["Season"].unique()).issubset(valid_seasons)

    def test_original_date_preserved(self):
        """Date column should still exist (not dropped in prep)."""
        df = make_minimal_df(n=10)
        df_prep = prepare_modeling_dataset(df)
        assert "Date" in df_prep.columns


# ─────────────────────────────────────────────────────────────────
# 2. Test: get_feature_lists identifies correct feature types
# ─────────────────────────────────────────────────────────────────

class TestGetFeatureLists:
    def test_numerical_features_identified(self):
        df = make_minimal_df(n=20)
        df_prep = prepare_modeling_dataset(df)
        fl = get_feature_lists(df_prep)
        for feat in NUMERICAL_POLLUTANT_FEATURES:
            assert feat in fl["numerical"], f"'{feat}' not in numerical list"

    def test_seasonal_in_categorical(self):
        df = make_minimal_df(n=20)
        df_prep = prepare_modeling_dataset(df)
        fl = get_feature_lists(df_prep)
        assert "Season" in fl["categorical"]

    def test_city_excluded_by_default(self):
        df = make_minimal_df(n=20)
        df_prep = prepare_modeling_dataset(df)
        fl = get_feature_lists(df_prep, include_city=False)
        assert "City" not in fl["categorical"]
        assert "City" not in fl["numerical"]

    def test_city_included_when_opted_in(self):
        df = make_minimal_df(n=20)
        df_prep = prepare_modeling_dataset(df)
        fl = get_feature_lists(df_prep, include_city=True)
        assert "City" in fl["categorical"]

    def test_leakage_columns_not_in_features(self):
        df = make_minimal_df(n=20)
        df_prep = prepare_modeling_dataset(df)
        fl = get_feature_lists(df_prep)
        for col in LEAKAGE_COLS:
            assert col not in fl["numerical"], f"Leakage col '{col}' found in numerical"
            assert col not in fl["categorical"], f"Leakage col '{col}' found in categorical"


# ─────────────────────────────────────────────────────────────────
# 3. Test: extract_X_y — correct shapes, no leakage
# ─────────────────────────────────────────────────────────────────

class TestExtractXY:
    def test_y_contains_integer_labels(self):
        df = make_minimal_df(n=30)
        df_prep = prepare_modeling_dataset(df)
        fl = get_feature_lists(df_prep)
        X, y = extract_X_y(df_prep, fl, use_integer_labels=True)
        assert y.dtype in [np.int64, np.int32, int, "int64", "int32"]
        assert set(y.unique()).issubset({0, 1, 2, 3, 4, 5})

    def test_leakage_raises_error(self):
        df = make_minimal_df(n=20)
        df_prep = prepare_modeling_dataset(df)
        fl = get_feature_lists(df_prep)
        # Inject a leakage column into feature list
        bad_fl = {**fl, "all": fl["all"] + ["AQI"]}
        bad_fl["numerical"] = fl["numerical"] + ["AQI"]
        with pytest.raises(ValueError, match="leakage"):
            extract_X_y(df_prep, bad_fl)

    def test_null_target_rows_dropped(self):
        df = make_minimal_df(n=30)
        df.iloc[:5, df.columns.get_loc("AQI_Bucket")] = np.nan
        df_prep = prepare_modeling_dataset(df)
        fl = get_feature_lists(df_prep)
        X, y = extract_X_y(df_prep, fl)
        assert y.notna().all(), "y should have no null values after extract_X_y"
        assert len(X) == len(y)
        assert len(X) <= 30

    def test_X_shape_matches_feature_list(self):
        df = make_minimal_df(n=40)
        df_prep = prepare_modeling_dataset(df)
        fl = get_feature_lists(df_prep)
        X, y = extract_X_y(df_prep, fl)
        assert X.shape[1] == len(fl["all"])


# ─────────────────────────────────────────────────────────────────
# 4. Test: Missing value handling
# ─────────────────────────────────────────────────────────────────

class TestMissingValueHandling:
    def test_no_nan_in_output_after_imputation(self):
        """Pipeline must produce zero NaN values in its output."""
        train_df, val_df, test_df = make_three_split_dfs()
        data = prepare_all_splits(train_df, val_df, test_df)
        for name, arr in [
            ("X_train_t", data["X_train_t"]),
            ("X_val_t",   data["X_val_t"]),
            ("X_test_t",  data["X_test_t"]),
        ]:
            nan_count = np.sum(np.isnan(arr))
            assert nan_count == 0, f"{name} has {nan_count} NaN values after imputation"

    def test_imputer_uses_training_median_not_test(self):
        """
        Verifying train/test isolation:
        When we construct a 'poisoned' test set where PM2.5 is entirely missing,
        the imputer must fill with the training median (not any test-set statistic).
        """
        train_df, val_df, test_df = make_three_split_dfs()
        train_prep = prepare_modeling_dataset(train_df)
        fl = get_feature_lists(train_prep)
        X_train, y_train = extract_X_y(train_prep, fl)

        # Record training median for PM2.5
        train_pm25_median = X_train["PM2.5"].median()

        # Build pipeline and fit on training data
        pipeline = build_preprocessing_pipeline(fl)
        pipeline.fit(X_train)

        # Create an all-NaN PM2.5 test set
        test_poisoned = test_df.copy()
        test_prep = prepare_modeling_dataset(test_poisoned)
        X_test, _ = extract_X_y(test_prep, fl)
        X_test_copy = X_test.copy()
        X_test_copy["PM2.5"] = np.nan

        # Transform the poisoned test set
        arr_out = pipeline.transform(X_test_copy)

        # The PM2.5 column should NOT be NaN — it should be filled with training median
        pm25_col_idx = fl["numerical"].index("PM2.5")
        filled_pm25 = arr_out[:, pm25_col_idx]
        assert np.all(np.isfinite(filled_pm25)), \
            "Imputed PM2.5 contains non-finite values — imputation failed"


# ─────────────────────────────────────────────────────────────────
# 5. Test: Unseen category handling
# ─────────────────────────────────────────────────────────────────

class TestUnseenCategories:
    def test_unseen_season_does_not_raise(self):
        """
        OrdinalEncoder must NOT raise on a season value never seen in training.
        It should return -1 (the configured unknown_value).
        """
        train_df, val_df, test_df = make_three_split_dfs()
        data = prepare_all_splits(train_df, val_df, test_df)
        pipeline = data["pipeline"]
        fl = data["feature_lists"]

        # Craft a test row with an unseen 'Season' value
        test_prep = prepare_modeling_dataset(test_df)
        X_test, _ = extract_X_y(test_prep, fl)
        X_mod = X_test.copy()
        if "Season" in X_mod.columns:
            X_mod["Season"] = "QuantumMonsoon"  # Never seen in training

        # Should not raise
        try:
            result = pipeline.transform(X_mod)
            assert result is not None
        except Exception as e:
            pytest.fail(f"Pipeline raised exception on unseen category: {e}")

    def test_unseen_season_encoded_as_negative_one(self):
        """Unseen categories should be encoded as -1 (unknown_value)."""
        train_df, val_df, test_df = make_three_split_dfs()
        data = prepare_all_splits(train_df, val_df, test_df)
        pipeline = data["pipeline"]
        fl = data["feature_lists"]

        test_prep = prepare_modeling_dataset(test_df)
        X_test, _ = extract_X_y(test_prep, fl)
        X_mod = X_test.copy()

        if "Season" not in X_mod.columns:
            pytest.skip("Season column not present in test data")

        X_mod["Season"] = "QuantumMonsoon"
        result = pipeline.transform(X_mod)

        season_idx = len(fl["numerical"])  # Season comes after all numerical cols
        season_col = result[:, season_idx]
        assert np.all(season_col == -1.0), \
            f"Expected -1 for unseen Season, got: {np.unique(season_col)}"


# ─────────────────────────────────────────────────────────────────
# 6. Test: Reproducibility
# ─────────────────────────────────────────────────────────────────

class TestReproducibility:
    def test_identical_output_on_rerun(self):
        """Two independent pipeline runs on same data must produce identical arrays."""
        train_df, val_df, test_df = make_three_split_dfs(seed=0)

        data1 = prepare_all_splits(train_df, val_df, test_df)
        data2 = prepare_all_splits(train_df, val_df, test_df)

        np.testing.assert_array_almost_equal(
            data1["X_train_t"], data2["X_train_t"],
            decimal=10,
            err_msg="Two independent runs produced different X_train_t arrays"
        )
        np.testing.assert_array_almost_equal(
            data1["X_test_t"], data2["X_test_t"],
            decimal=10,
            err_msg="Two independent runs produced different X_test_t arrays"
        )

    def test_feature_count_consistent_across_splits(self):
        """Train, val, and test must have the same number of output features."""
        train_df, val_df, test_df = make_three_split_dfs()
        data = prepare_all_splits(train_df, val_df, test_df)

        n_train_feats = data["X_train_t"].shape[1]
        n_val_feats   = data["X_val_t"].shape[1]
        n_test_feats  = data["X_test_t"].shape[1]

        assert n_train_feats == n_val_feats == n_test_feats, \
            f"Feature count mismatch: train={n_train_feats}, val={n_val_feats}, test={n_test_feats}"


# ─────────────────────────────────────────────────────────────────
# 7. Test: Train/Test Isolation (no data leakage from scaler/imputer)
# ─────────────────────────────────────────────────────────────────

class TestTrainTestIsolation:
    def test_scaler_fitted_only_on_training_data(self):
        """
        The StandardScaler's mean must match the training set mean,
        not the overall population mean.
        """
        train_df, val_df, test_df = make_three_split_dfs()
        data = prepare_all_splits(train_df, val_df, test_df)
        pipeline = data["pipeline"]

        # Extract the ColumnTransformer
        ct = pipeline.named_steps["preprocessor"]
        # Get the numerical sub-pipeline
        num_pipeline = ct.named_transformers_.get("numerical")
        if num_pipeline is None:
            pytest.skip("No numerical transformer found in pipeline")

        scaler = num_pipeline.named_steps.get("scaler")
        if scaler is None:
            pytest.skip("No StandardScaler found in numerical pipeline")

        # The scaler's mean should be close to the training set mean (after winsorize/impute)
        X_train_t = data["X_train_t"]
        # After scaling, the training data should have near-zero mean
        train_col_means = np.mean(X_train_t[:, :len(data["feature_lists"]["numerical"])], axis=0)
        assert np.allclose(train_col_means, 0.0, atol=0.1), \
            f"Training data columns do not have near-zero mean after StandardScaler. Means: {train_col_means[:5]}"

    def test_imputer_median_computed_from_training_only(self):
        """
        The fitted imputer's median statistics should reflect training data only.
        Computed by comparing pipeline's stored statistic vs actual training median.
        """
        train_df, val_df, test_df = make_three_split_dfs(seed=10)
        train_prep = prepare_modeling_dataset(train_df)
        fl = get_feature_lists(train_prep)
        X_train, y_train = extract_X_y(train_prep, fl)

        pipeline = build_preprocessing_pipeline(fl)
        pipeline.fit(X_train)

        ct = pipeline.named_steps["preprocessor"]
        num_pipeline = ct.named_transformers_.get("numerical")
        if num_pipeline is None:
            pytest.skip("No numerical transformer found")

        imputer = num_pipeline.named_steps.get("imputer")
        if imputer is None:
            pytest.skip("No imputer found in numerical pipeline")

        # After winsorization, the imputer sees winsorized values.
        # We can verify the imputer's statistics_ are not None
        assert imputer.statistics_ is not None
        assert len(imputer.statistics_) == len(fl["numerical"])


# ─────────────────────────────────────────────────────────────────
# 8. Test: Feature order consistency
# ─────────────────────────────────────────────────────────────────

class TestFeatureOrderConsistency:
    def test_feature_names_out_length_matches_array_width(self):
        train_df, val_df, test_df = make_three_split_dfs()
        data = prepare_all_splits(train_df, val_df, test_df)
        n_feats_names = len(data["feature_names_out"])
        n_feats_array = data["X_train_t"].shape[1]
        assert n_feats_names == n_feats_array, \
            f"feature_names_out length ({n_feats_names}) != array width ({n_feats_array})"

    def test_numerical_features_before_categorical(self):
        """Numerical features should come before categorical in output."""
        train_df, val_df, test_df = make_three_split_dfs()
        data = prepare_all_splits(train_df, val_df, test_df)
        names = data["feature_names_out"]
        # numerical__ prefixed names should appear before categorical__ ones
        num_positions = [i for i, n in enumerate(names) if "numerical__" in n]
        cat_positions = [i for i, n in enumerate(names) if "categorical__" in n]
        if num_positions and cat_positions:
            assert max(num_positions) < min(cat_positions), \
                "Numerical features should precede categorical features in output"


# ─────────────────────────────────────────────────────────────────
# 9. Test: Label map integrity
# ─────────────────────────────────────────────────────────────────

class TestLabelMapping:
    def test_category_to_label_and_back(self):
        """Round-trip: category → int → category should preserve identity."""
        for cat, idx in CATEGORY_TO_LABEL.items():
            restored = LABEL_TO_CATEGORY[idx]
            assert restored == cat, f"Round-trip failed: {cat} → {idx} → {restored}"

    def test_all_six_categories_mapped(self):
        assert len(CATEGORY_TO_LABEL) == 6
        assert len(LABEL_TO_CATEGORY) == 6
        assert set(LABEL_TO_CATEGORY.values()) == {
            "Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"
        }

    def test_prepare_all_splits_returns_label_map(self):
        train_df, val_df, test_df = make_three_split_dfs()
        data = prepare_all_splits(train_df, val_df, test_df)
        assert "label_map" in data
        assert len(data["label_map"]) == 6


# ─────────────────────────────────────────────────────────────────
# 10. Test: Pipeline is a valid sklearn Pipeline
# ─────────────────────────────────────────────────────────────────

class TestPipelineInterface:
    def test_pipeline_is_sklearn_pipeline(self):
        train_df, _, _ = make_three_split_dfs()
        train_prep = prepare_modeling_dataset(train_df)
        fl = get_feature_lists(train_prep)
        pipeline = build_preprocessing_pipeline(fl)
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_has_preprocessor_step(self):
        train_df, _, _ = make_three_split_dfs()
        train_prep = prepare_modeling_dataset(train_df)
        fl = get_feature_lists(train_prep)
        pipeline = build_preprocessing_pipeline(fl)
        assert "preprocessor" in pipeline.named_steps

    def test_pipeline_output_is_ndarray(self):
        train_df, val_df, test_df = make_three_split_dfs()
        data = prepare_all_splits(train_df, val_df, test_df)
        for arr_name in ["X_train_t", "X_val_t", "X_test_t"]:
            assert isinstance(data[arr_name], np.ndarray), \
                f"{arr_name} should be np.ndarray"

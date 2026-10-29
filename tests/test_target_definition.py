"""
Unit Tests for AirSense Target Definition, Leakage Prevention & Data Splitting
==============================================================================
Tests CPCB label validation, threshold derivation, leakage detection,
and multiple splitting strategies (stratified, temporal, spatial).
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_loader import load_city_day, CPCB_AQI_CATEGORIES
from src.target_definition import (
    aqi_to_cpcb_bucket,
    validate_target_labels,
    identify_leakage_features,
    prepare_feature_target_matrices,
    LEAKAGE_COLUMNS
)
from src.data_split import (
    stratified_data_split,
    temporal_data_split,
    spatial_data_split,
    audit_data_split
)


def test_aqi_to_cpcb_bucket_breakpoints():
    """Verify exact CPCB NAQI threshold mappings."""
    # Good (0-50)
    assert aqi_to_cpcb_bucket(0.0) == "Good"
    assert aqi_to_cpcb_bucket(25.0) == "Good"
    assert aqi_to_cpcb_bucket(50.0) == "Good"

    # Satisfactory (51-100)
    assert aqi_to_cpcb_bucket(51.0) == "Satisfactory"
    assert aqi_to_cpcb_bucket(100.0) == "Satisfactory"

    # Moderate (101-200)
    assert aqi_to_cpcb_bucket(101.0) == "Moderate"
    assert aqi_to_cpcb_bucket(200.0) == "Moderate"

    # Poor (201-300)
    assert aqi_to_cpcb_bucket(201.0) == "Poor"
    assert aqi_to_cpcb_bucket(300.0) == "Poor"

    # Very Poor (301-400)
    assert aqi_to_cpcb_bucket(301.0) == "Very Poor"
    assert aqi_to_cpcb_bucket(400.0) == "Very Poor"

    # Severe (401+)
    assert aqi_to_cpcb_bucket(401.0) == "Severe"
    assert aqi_to_cpcb_bucket(2049.0) == "Severe"

    # Invalid / Negative / NaN
    assert aqi_to_cpcb_bucket(-5.0) is None
    assert aqi_to_cpcb_bucket(np.nan) is None


def test_validate_target_labels_on_dataset():
    """Verify target validation on actual city_day.csv."""
    df = load_city_day(drop_na_target=False)
    audit = validate_target_labels(df, target_col="AQI_Bucket")

    assert audit["total_rows"] == 29531
    assert audit["valid_rows"] == 24850
    assert audit["null_rows"] == 4681
    assert audit["is_valid_standard"] is True
    assert len(audit["invalid_labels"]) == 0
    assert audit["aqi_bucket_mismatches"] == 0  # 100% agreement with continuous AQI thresholds


def test_identify_leakage_features():
    """Verify that columns containing target information are flagged as leakage."""
    candidates = ["PM2.5", "PM10", "NO2", "AQI", "AQI_Bucket", "Target_Label", "Month", "City"]
    leakage = identify_leakage_features(candidates)

    assert "AQI" in leakage
    assert "AQI_Bucket" in leakage
    assert "Target_Label" in leakage
    assert "PM2.5" not in leakage
    assert "NO2" not in leakage
    assert "City" not in leakage


def test_prepare_feature_target_matrices_leakage_free():
    """Verify that X contains 0 leakage features and matches y length."""
    df = load_city_day(drop_na_target=True)
    X, y, feature_names = prepare_feature_target_matrices(df, target_col="AQI_Bucket")

    assert len(X) == len(df)
    assert len(y) == len(df)
    assert "AQI" not in X.columns
    assert "AQI_Bucket" not in X.columns
    assert "Target_Label" not in X.columns
    assert "AQI" not in feature_names

    # All criteria pollutants should be present
    assert "PM2.5" in feature_names
    assert "PM10" in feature_names
    assert "CO" in feature_names


def test_stratified_data_split_integrity():
    """Verify stratified split balance, exact row counts, and zero index overlap."""
    df = load_city_day(drop_na_target=True)
    train_df, val_df, test_df = stratified_data_split(df, target_col="AQI_Bucket", test_size=0.15, val_size=0.15, random_state=42)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    audit = audit_data_split(train_df, val_df, test_df, target_col="AQI_Bucket")
    assert audit["train_pct"] >= 69.5 and audit["train_pct"] <= 70.5
    assert audit["val_pct"] >= 14.5 and audit["val_pct"] <= 15.5
    assert audit["test_pct"] >= 14.5 and audit["test_pct"] <= 15.5

    # Check each class distribution
    for cat in CPCB_AQI_CATEGORIES:
        tr_pct = audit["class_distributions"]["train"][cat]
        te_pct = audit["class_distributions"]["test"][cat]
        assert abs(tr_pct - te_pct) < 1.5


def test_temporal_data_split_integrity():
    """Verify temporal split chronological ordering and non-overlap."""
    df = load_city_day(drop_na_target=True, parse_dates=True)
    train_df, val_df, test_df = temporal_data_split(
        df,
        date_col="Date",
        train_end="2019-06-30",
        val_end="2019-12-31"
    )

    assert train_df["Date"].max() <= pd.to_datetime("2019-06-30")
    assert val_df["Date"].min() > pd.to_datetime("2019-06-30")
    assert val_df["Date"].max() <= pd.to_datetime("2019-12-31")
    assert test_df["Date"].min() > pd.to_datetime("2019-12-31")

    assert len(train_df) + len(val_df) + len(test_df) == len(df)


def test_spatial_data_split_integrity():
    """Verify spatial split ensures disjoint sets of cities."""
    df = load_city_day(drop_na_target=True)
    train_df, val_df, test_df = spatial_data_split(df, group_col="City", random_state=42)

    train_cities = set(train_df["City"].unique())
    val_cities = set(val_df["City"].unique())
    test_cities = set(test_df["City"].unique())

    assert len(train_cities.intersection(val_cities)) == 0
    assert len(train_cities.intersection(test_cities)) == 0
    assert len(val_cities.intersection(test_cities)) == 0

    total_cities = len(train_cities) + len(val_cities) + len(test_cities)
    assert total_cities == df["City"].nunique()


"""
Unit Tests for AirSense Preprocessing Module
============================================
Tests data filtering, temporal feature extraction, stratified splitting,
leakage-free imputation, and target encoding.
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

from src.data_loader import load_city_day, POLLUTANT_COLUMNS, CPCB_AQI_CATEGORIES
from src.preprocessing import (
    filter_labeled_data,
    extract_temporal_features,
    create_stratified_split,
    fit_imputer,
    transform_impute,
    encode_target,
    CATEGORY_TO_LABEL
)


def test_filter_labeled_data():
    """Verify that filter_labeled_data correctly removes rows missing the target."""
    raw_df = load_city_day(parse_dates=False, drop_na_target=False)
    assert len(raw_df) == 29531

    filtered_df, audit = filter_labeled_data(raw_df, target_col="AQI_Bucket")
    assert audit["initial_rows"] == 29531
    assert audit["dropped_missing_target"] == 4681
    assert audit["retained_rows"] == 24850
    assert len(filtered_df) == 24850
    assert filtered_df["AQI_Bucket"].isnull().sum() == 0


def test_extract_temporal_features():
    """Verify that temporal features are accurately extracted from the Date column."""
    sample_df = pd.DataFrame({
        "Date": ["2018-01-15", "2019-06-20", "2020-10-05"],
        "AQI_Bucket": ["Moderate", "Satisfactory", "Poor"]
    })
    enriched_df = extract_temporal_features(sample_df)

    assert "Year" in enriched_df.columns
    assert "Month" in enriched_df.columns
    assert "Day" in enriched_df.columns
    assert "DayOfWeek" in enriched_df.columns
    assert "IsWeekend" in enriched_df.columns
    assert "Season" in enriched_df.columns

    assert enriched_df.loc[0, "Year"] == 2018
    assert enriched_df.loc[0, "Month"] == 1
    assert enriched_df.loc[0, "Season"] == "Winter"
    assert enriched_df.loc[1, "Season"] == "Monsoon"
    assert enriched_df.loc[2, "Season"] == "Post-Monsoon"


def test_create_stratified_split():
    """Verify that stratified splitting preserves class distributions and has 0 overlap."""
    raw_df = load_city_day(drop_na_target=True)
    train_df, val_df, test_df = create_stratified_split(
        raw_df,
        target_col="AQI_Bucket",
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )

    total_len = len(train_df) + len(val_df) + len(test_df)
    assert total_len == len(raw_df), "Split row count does not match input row count"

    # Verify no index collision
    train_dates_cities = set(zip(train_df["City"], train_df["Date"]))
    val_dates_cities = set(zip(val_df["City"], val_df["Date"]))
    test_dates_cities = set(zip(test_df["City"], test_df["Date"]))

    assert len(train_dates_cities.intersection(val_dates_cities)) == 0
    assert len(train_dates_cities.intersection(test_dates_cities)) == 0
    assert len(val_dates_cities.intersection(test_dates_cities)) == 0

    # Verify proportions for all 6 categories
    for cat in CPCB_AQI_CATEGORIES:
        full_pct = (raw_df["AQI_Bucket"] == cat).mean()
        train_pct = (train_df["AQI_Bucket"] == cat).mean()
        val_pct = (val_df["AQI_Bucket"] == cat).mean()
        test_pct = (test_df["AQI_Bucket"] == cat).mean()

        # Proportions should match within 1% absolute margin
        assert abs(full_pct - train_pct) < 0.015, f"Train skew for {cat}"
        assert abs(full_pct - val_pct) < 0.02, f"Val skew for {cat}"
        assert abs(full_pct - test_pct) < 0.02, f"Test skew for {cat}"


def test_fit_and_transform_impute():
    """Verify that imputer fits strictly on train and leaves 0 nulls after transform."""
    raw_df = load_city_day(drop_na_target=True)
    train_df, val_df, test_df = create_stratified_split(raw_df, random_state=42)

    imputer_params = fit_imputer(train_df, feature_cols=POLLUTANT_COLUMNS, group_col="City")

    # Target column must NOT be present in imputer params
    assert "AQI" not in imputer_params["global_medians"]
    assert "AQI_Bucket" not in imputer_params["global_medians"]

    # Transform all three splits
    train_imp = transform_impute(train_df, imputer_params)
    val_imp = transform_impute(val_df, imputer_params)
    test_imp = transform_impute(test_df, imputer_params)

    for col in POLLUTANT_COLUMNS:
        assert train_imp[col].isnull().sum() == 0, f"Nulls remain in train_imp for {col}"
        assert val_imp[col].isnull().sum() == 0, f"Nulls remain in val_imp for {col}"
        assert test_imp[col].isnull().sum() == 0, f"Nulls remain in test_imp for {col}"


def test_encode_target():
    """Verify that ordinal CPCB category encoding maps accurately to 0-5 integers."""
    sample_df = pd.DataFrame({"AQI_Bucket": CPCB_AQI_CATEGORIES})
    encoded_df = encode_target(sample_df)

    assert "Target_Label" in encoded_df.columns
    expected_labels = [CATEGORY_TO_LABEL[c] for c in CPCB_AQI_CATEGORIES]
    assert list(encoded_df["Target_Label"]) == expected_labels
    assert list(encoded_df["Target_Label"]) == [0, 1, 2, 3, 4, 5]


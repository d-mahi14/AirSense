"""
Unit and Integration Tests for AirSense Data Loader
===================================================
Tests directory resolution, file discovery, data loading, schema integrity,
and CPCB standard compliance.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_loader import (
    get_data_dir,
    discover_raw_files,
    load_stations,
    load_city_day,
    load_city_hour,
    load_station_day,
    load_station_hour,
    load_dataset,
    POLLUTANT_COLUMNS,
    TARGET_COLUMNS,
    CPCB_AQI_CATEGORIES,
    WEATHER_COLUMNS
)


def test_get_data_dir_default():
    """Verify that get_data_dir discovers the raw directory automatically."""
    data_dir = get_data_dir()
    assert data_dir.exists(), "Discovered data directory does not exist"
    assert data_dir.is_dir(), "Discovered path is not a directory"
    assert (data_dir / "city_day.csv").exists(), "city_day.csv not found in discovered directory"


def test_get_data_dir_custom_path():
    """Verify that a valid custom path is resolved and validated."""
    actual_dir = get_data_dir()
    custom_dir = get_data_dir(str(actual_dir))
    assert custom_dir == actual_dir


def test_get_data_dir_invalid_path():
    """Verify that FileNotFoundError is raised when an invalid directory is provided."""
    with pytest.raises(FileNotFoundError):
        get_data_dir("C:/non_existent_airsense_directory_12345")


def test_discover_raw_files():
    """Verify that all 5 canonical CSV files are discovered on disk."""
    files = discover_raw_files()
    expected_keys = {"stations", "city_day", "city_hour", "station_day", "station_hour"}
    assert set(files.keys()) == expected_keys

    for key, path in files.items():
        assert path.exists(), f"Discovered file '{key}' at {path} does not exist"
        assert path.is_file(), f"Discovered path '{path}' is not a regular file"
        assert path.suffix == ".csv", f"Discovered file '{path}' is not a CSV"


def test_load_stations():
    """Verify that stations.csv loads cleanly with the expected schema."""
    df = load_stations()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 230, f"Expected 230 stations, found {len(df)}"
    expected_cols = {"StationId", "StationName", "City", "State", "Status"}
    assert set(df.columns) == expected_cols
    assert df["StationId"].isnull().sum() == 0, "StationId contains nulls"
    assert df["StationId"].nunique() == 230, "StationId is not unique in stations.csv"


def test_load_city_day():
    """Verify loading of city_day.csv with date parsing and target dropping."""
    df = load_city_day(parse_dates=True, drop_na_target=False)
    assert len(df) == 29531, f"Expected 29531 rows, found {len(df)}"
    assert pd.api.types.is_datetime64_any_dtype(df["Date"]), "Date column was not parsed as datetime"

    # Verify all pollutants exist
    for pollutant in POLLUTANT_COLUMNS:
        assert pollutant in df.columns, f"Missing pollutant column: {pollutant}"

    # Verify target columns
    assert "AQI" in df.columns
    assert "AQI_Bucket" in df.columns

    # Verify drop_na_target functionality
    df_clean = load_city_day(drop_na_target=True)
    assert len(df_clean) == 24850, f"Expected 24850 valid rows, found {len(df_clean)}"
    assert df_clean["AQI_Bucket"].isnull().sum() == 0
    assert df_clean["AQI"].isnull().sum() == 0


def test_load_city_hour_sample():
    """Verify loading of a sample from city_hour.csv."""
    df_sample = load_city_hour(nrows=50, parse_dates=True)
    assert len(df_sample) == 50
    assert pd.api.types.is_datetime64_any_dtype(df_sample["Datetime"])
    assert "City" in df_sample.columns
    assert "PM2.5" in df_sample.columns


def test_load_station_day_sample():
    """Verify loading of station_day.csv."""
    df = load_station_day(parse_dates=False)
    assert len(df) == 108035
    assert "StationId" in df.columns
    assert "Date" in df.columns


def test_load_station_hour_sample():
    """Verify loading a small slice of station_hour.csv without memory strain."""
    df = load_station_hour(nrows=100)
    assert len(df) == 100
    assert "StationId" in df.columns
    assert "Datetime" in df.columns


def test_load_dataset_dispatcher():
    """Verify the generic loader dispatcher."""
    stations_df = load_dataset("stations")
    assert len(stations_df) == 230

    with pytest.raises(ValueError):
        load_dataset("unknown_dataset_name")


def test_weather_variables_absence():
    """Explicitly verify that WEATHER_COLUMNS is empty and no weather columns exist."""
    assert len(WEATHER_COLUMNS) == 0, "WEATHER_COLUMNS must be empty"
    df = load_city_day(drop_na_target=False)
    weather_keywords = ["temp", "humidity", "rh", "wind", "rain", "precip", "solar", "pressure"]
    found_weather = [c for c in df.columns if any(kw in c.lower() for kw in weather_keywords)]
    assert len(found_weather) == 0, f"Unexpected weather columns detected: {found_weather}"


def test_cpcb_aqi_categories():
    """Verify that city_day categories strictly belong to official CPCB categories."""
    df = load_city_day(drop_na_target=True)
    unique_buckets = set(df["AQI_Bucket"].unique())
    expected_categories = set(CPCB_AQI_CATEGORIES)
    assert unique_buckets.issubset(expected_categories), f"Unexpected categories: {unique_buckets - expected_categories}"

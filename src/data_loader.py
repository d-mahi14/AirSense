"""
AirSense Data Loader Module
===========================
Provides configurable, robust data discovery and loading for the
India Air Quality dataset (2015-2020).

Key features:
- Discovers the raw CSV files using configurable paths, environment variables,
  or standard default candidate directories (handling 'archive (3)' vs 'archive(3)').
- Specific loader functions for each of the 5 files with appropriate data types,
  date parsing, and optional chunking for large files.
- Preserves raw files strictly as read-only.
- Explicitly documents known schemas, pollutant criteria, and the absence of weather fields.
"""

from pathlib import Path
import os
from typing import Dict, Optional, Union, Generator, List
import pandas as pd

# Core domain constants
POLLUTANT_COLUMNS: List[str] = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3", "Benzene",
    "Toluene", "Xylene"
]

TARGET_COLUMNS: List[str] = ["AQI", "AQI_Bucket"]

CPCB_AQI_CATEGORIES: List[str] = [
    "Good",
    "Satisfactory",
    "Moderate",
    "Poor",
    "Very Poor",
    "Severe"
]

# Explicit constant documenting absence of weather measurements in this dataset
WEATHER_COLUMNS: List[str] = []

# Canonical expected filenames
RAW_FILENAMES: Dict[str, str] = {
    "stations": "stations.csv",
    "city_day": "city_day.csv",
    "city_hour": "city_hour.csv",
    "station_day": "station_day.csv",
    "station_hour": "station_hour.csv"
}


def get_data_dir(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve and validate the raw data directory path.

    Search priority:
    1. custom_path parameter (if provided, strictly validated)
    2. AIRSENSE_DATA_DIR environment variable
    3. Standard relative and absolute workspace paths:
       - 'data/raw/archive (3)'
       - 'data/raw/archive(3)'
       - 'data/raw'
       - Project root relative paths

    Returns:
        Path: Validated pathlib.Path pointing to the directory containing raw CSVs.

    Raises:
        FileNotFoundError: If no valid directory containing the raw CSVs is found.
    """
    if custom_path:
        p = Path(custom_path)
        if p.exists() and p.is_dir():
            if (p / "city_day.csv").exists() or (p / "stations.csv").exists():
                return p.resolve()
        raise FileNotFoundError(
            f"The provided custom data directory does not exist or does not contain expected raw files: {custom_path}"
        )

    candidates = []

    env_dir = os.environ.get("AIRSENSE_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    # Determine script and project locations
    current_dir = Path.cwd()
    project_root = Path(__file__).resolve().parent.parent if "__file__" in globals() else current_dir

    standard_relative_paths = [
        current_dir / "data" / "raw" / "archive (3)",
        current_dir / "data" / "raw" / "archive(3)",
        current_dir / "data" / "raw",
        project_root / "data" / "raw" / "archive (3)",
        project_root / "data" / "raw" / "archive(3)",
        project_root / "data" / "raw",
    ]
    candidates.extend(standard_relative_paths)

    # Check each candidate
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            # Verify it contains at least one of our key CSV files
            if (candidate / "city_day.csv").exists() or (candidate / "stations.csv").exists():
                return candidate.resolve()

    raise FileNotFoundError(
        f"Could not locate the raw data directory. Checked paths:\n" +
        "\n".join(f"  - {p.resolve() if hasattr(p, 'resolve') else p}" for p in candidates) +
        "\nPlease provide a valid custom_path or set the AIRSENSE_DATA_DIR environment variable."
    )


def discover_raw_files(data_dir: Optional[Union[str, Path]] = None) -> Dict[str, Path]:
    """
    Discover all 5 raw dataset CSV files in the data directory.

    Returns:
        Dict[str, Path]: Dictionary mapping canonical keys ('stations', 'city_day', etc.)
                         to their absolute Path objects.

    Raises:
        FileNotFoundError: If any of the required files are missing.
    """
    resolved_dir = get_data_dir(data_dir)
    found_files: Dict[str, Path] = {}

    for key, filename in RAW_FILENAMES.items():
        file_path = resolved_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing expected raw dataset file '{filename}' in {resolved_dir}"
            )
        found_files[key] = file_path.resolve()

    return found_files


def load_stations(data_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load the monitoring stations metadata table (stations.csv).
    """
    files = discover_raw_files(data_dir)
    return pd.read_csv(files["stations"])


def load_city_day(
    data_dir: Optional[Union[str, Path]] = None,
    parse_dates: bool = True,
    drop_na_target: bool = False
) -> pd.DataFrame:
    """
    Load daily city-level air quality observations (city_day.csv).

    Args:
        data_dir: Directory containing raw CSV files (optional).
        parse_dates: Whether to parse the 'Date' column as datetime.
        drop_na_target: Whether to drop rows where 'AQI_Bucket' or 'AQI' is missing.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    files = discover_raw_files(data_dir)
    date_cols = ["Date"] if parse_dates else None
    df = pd.read_csv(files["city_day"], parse_dates=date_cols)

    if drop_na_target:
        df = df.dropna(subset=["AQI_Bucket"]).reset_index(drop=True)

    return df


def load_city_hour(
    data_dir: Optional[Union[str, Path]] = None,
    parse_dates: bool = True,
    nrows: Optional[int] = None,
    drop_na_target: bool = False
) -> pd.DataFrame:
    """
    Load hourly city-level observations (city_hour.csv).
    """
    files = discover_raw_files(data_dir)
    date_cols = ["Datetime"] if parse_dates else None
    df = pd.read_csv(files["city_hour"], parse_dates=date_cols, nrows=nrows)

    if drop_na_target:
        df = df.dropna(subset=["AQI_Bucket"]).reset_index(drop=True)

    return df


def load_station_day(
    data_dir: Optional[Union[str, Path]] = None,
    parse_dates: bool = True,
    drop_na_target: bool = False
) -> pd.DataFrame:
    """
    Load daily station-level observations (station_day.csv).
    """
    files = discover_raw_files(data_dir)
    date_cols = ["Date"] if parse_dates else None
    df = pd.read_csv(files["station_day"], parse_dates=date_cols)

    if drop_na_target:
        df = df.dropna(subset=["AQI_Bucket"]).reset_index(drop=True)

    return df


def load_station_hour(
    data_dir: Optional[Union[str, Path]] = None,
    parse_dates: bool = False,
    nrows: Optional[int] = None,
    chunksize: Optional[int] = None
) -> Union[pd.DataFrame, Generator[pd.DataFrame, None, None]]:
    """
    Load hourly station-level observations (station_hour.csv).

    Due to large file size (209.5 MB, ~2.59M rows), chunksize or nrows
    can be used to conserve memory.
    """
    files = discover_raw_files(data_dir)
    date_cols = ["Datetime"] if parse_dates else None
    dtype_spec = {"AQI_Bucket": str, "StationId": str}

    return pd.read_csv(
        files["station_hour"],
        parse_dates=date_cols,
        nrows=nrows,
        chunksize=chunksize,
        dtype=dtype_spec,
        low_memory=False
    )


def load_dataset(
    name: str = "city_day",
    data_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Generic loader dispatcher for any of the 5 AirSense raw datasets.

    Args:
        name: One of ['stations', 'city_day', 'city_hour', 'station_day', 'station_hour'].
        data_dir: Path to directory containing raw CSVs.
        **kwargs: Arguments passed to specific loaders.
    """
    loaders = {
        "stations": load_stations,
        "city_day": load_city_day,
        "city_hour": load_city_hour,
        "station_day": load_station_day,
        "station_hour": load_station_hour
    }

    if name not in loaders:
        raise ValueError(f"Unknown dataset name '{name}'. Expected one of {list(loaders.keys())}")

    return loaders[name](data_dir=data_dir, **kwargs)

"""
AirSense - Air Quality Prediction and Analysis Using Machine Learning
=====================================================================
Core source package.
"""

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

from src.preprocessing import (
    filter_labeled_data,
    extract_temporal_features,
    create_stratified_split,
    fit_imputer,
    transform_impute,
    encode_target,
    save_processed_splits,
    CATEGORY_TO_LABEL,
    LABEL_TO_CATEGORY,
    SEASON_MAP
)

from src.target_definition import (
    aqi_to_cpcb_bucket,
    validate_target_labels,
    identify_leakage_features,
    prepare_feature_target_matrices,
    CPCB_AQI_BREAKPOINTS,
    LEAKAGE_COLUMNS
)

from src.data_split import (
    stratified_data_split,
    temporal_data_split,
    spatial_data_split,
    audit_data_split,
    save_split_datasets
)

__all__ = [
    "get_data_dir",
    "discover_raw_files",
    "load_stations",
    "load_city_day",
    "load_city_hour",
    "load_station_day",
    "load_station_hour",
    "load_dataset",
    "POLLUTANT_COLUMNS",
    "TARGET_COLUMNS",
    "CPCB_AQI_CATEGORIES",
    "WEATHER_COLUMNS",
    "filter_labeled_data",
    "extract_temporal_features",
    "create_stratified_split",
    "fit_imputer",
    "transform_impute",
    "encode_target",
    "save_processed_splits",
    "CATEGORY_TO_LABEL",
    "LABEL_TO_CATEGORY",
    "SEASON_MAP",
    "aqi_to_cpcb_bucket",
    "validate_target_labels",
    "identify_leakage_features",
    "prepare_feature_target_matrices",
    "CPCB_AQI_BREAKPOINTS",
    "LEAKAGE_COLUMNS",
    "stratified_data_split",
    "temporal_data_split",
    "spatial_data_split",
    "audit_data_split",
    "save_split_datasets"
]


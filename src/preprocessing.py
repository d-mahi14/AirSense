"""
AirSense Preprocessing & Data Cleaning Module
=============================================
Provides disciplined, leak-free preprocessing transformations for the
AirSense air quality classification benchmark.

Key Design Principles & Constraints:
1. Target Integrity: Never impute missing target labels (AQI_Bucket or AQI).
   Unlabeled observations are filtered out and documented.
2. Leakage Prevention: Imputation statistics (medians) and scaling parameters
   are fitted strictly on training data and subsequently applied to validation
   and test splits.
3. Feature Independence: Predictor features are NEVER imputed using target labels.
4. Temporal Safety: Calendar features (Year, Month, Day, DayOfWeek, Season)
   are extracted deterministically from the Date timestamp.
5. Immutability: Raw CSV datasets remain 100% read-only.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np

from src.data_loader import (
    POLLUTANT_COLUMNS,
    TARGET_COLUMNS,
    CPCB_AQI_CATEGORIES
)

# Standard Indian meteorological season definitions
SEASON_MAP: Dict[int, str] = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon"
}

# Ordinal mapping of CPCB categories to integers (0 to 5)
CATEGORY_TO_LABEL: Dict[str, int] = {
    category: idx for idx, category in enumerate(CPCB_AQI_CATEGORIES)
}
LABEL_TO_CATEGORY: Dict[int, str] = {
    idx: category for idx, category in enumerate(CPCB_AQI_CATEGORIES)
}


def filter_labeled_data(df: pd.DataFrame, target_col: str = "AQI_Bucket") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Filter observations that contain ground-truth target labels.

    Strict Rule: Target labels are NEVER imputed. Rows missing the target
    cannot be reliably evaluated for supervised classification under CPCB
    regulatory criteria and are segregated.

    Args:
        df: Input DataFrame (e.g., loaded city_day.csv).
        target_col: Name of target column (default 'AQI_Bucket').

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned DataFrame with valid targets,
        and an audit summary dictionary recording dropped counts.
    """
    total_rows = len(df)
    missing_target_mask = df[target_col].isnull()
    dropped_count = int(missing_target_mask.sum())
    retained_df = df[~missing_target_mask].copy().reset_index(drop=True)

    audit_summary = {
        "initial_rows": total_rows,
        "dropped_missing_target": dropped_count,
        "dropped_pct": round((dropped_count / total_rows) * 100, 2) if total_rows > 0 else 0.0,
        "retained_rows": len(retained_df),
        "target_col": target_col
    }

    return retained_df, audit_summary


def extract_temporal_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Extract safe temporal and calendar covariates from the Date column.

    Features generated:
    - Year (integer, e.g., 2015-2020)
    - Month (1 to 12)
    - Day (1 to 31)
    - DayOfWeek (0=Monday to 6=Sunday)
    - IsWeekend (0 or 1)
    - Season (Winter, Summer, Monsoon, Post-Monsoon)

    Args:
        df: Input DataFrame containing a valid date column.
        date_col: Name of the date column.

    Returns:
        pd.DataFrame: DataFrame with additional temporal columns.
    """
    df_out = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_out[date_col]):
        df_out[date_col] = pd.to_datetime(df_out[date_col])

    dt = df_out[date_col].dt
    df_out["Year"] = dt.year
    df_out["Month"] = dt.month
    df_out["Day"] = dt.day
    df_out["DayOfWeek"] = dt.dayofweek
    df_out["IsWeekend"] = (dt.dayofweek >= 5).astype(int)
    df_out["Season"] = dt.month.map(SEASON_MAP)

    return df_out


def create_stratified_split(
    df: pd.DataFrame,
    target_col: str = "AQI_Bucket",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partition the labeled dataset into stratified Train, Validation, and Test splits.
    Preserves exact class proportions across all six CPCB AQI categories.

    Uses pure pandas/numpy random sampling to guarantee reproducibility without
    hard external library dependencies.

    Split default: 70% Train, 15% Validation, 15% Test.

    Args:
        df: Cleaned DataFrame with valid target labels.
        target_col: Categorical column to stratify over.
        test_size: Fraction of data reserved for held-out testing (e.g. 0.15).
        val_size: Fraction of data reserved for validation / tuning (e.g. 0.15).
        random_state: Seed for random number generation.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df).
    """
    rng = np.random.RandomState(random_state)
    train_indices = []
    val_indices = []
    test_indices = []

    for category, group in df.groupby(target_col):
        indices = group.index.values.copy()
        rng.shuffle(indices)

        n_group = len(indices)
        n_test = int(round(n_group * test_size))
        n_val = int(round(n_group * val_size))

        test_idx = indices[:n_test]
        val_idx = indices[n_test : n_test + n_val]
        train_idx = indices[n_test + n_val :]

        test_indices.extend(test_idx)
        val_indices.extend(val_idx)
        train_indices.extend(train_idx)

    train_df = df.loc[train_indices].copy().sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    val_df = df.loc[val_indices].copy().sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test_df = df.loc[test_indices].copy().sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return train_df, val_df, test_df


def fit_imputer(
    train_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    group_col: Optional[str] = "City"
) -> Dict[str, Any]:
    """
    Fit missing-value imputation parameters strictly on the training set.

    Strategy:
    1. Primary: Grouped median by City for each pollutant.
    2. Fallback: Global training median for any pollutant if a city has no records for that pollutant.

    Guarantees:
    - Target column is NEVER used in calculating feature medians.
    - Zero data leakage into validation or test sets.

    Args:
        train_df: Training DataFrame.
        feature_cols: List of numerical pollutant features to impute.
        group_col: Grouping column for localized medians (default 'City').

    Returns:
        Dict[str, Any]: Dictionary containing learned group medians and global medians.
    """
    if feature_cols is None:
        feature_cols = [c for c in POLLUTANT_COLUMNS if c in train_df.columns]

    global_medians = {}
    grouped_medians = {}

    for col in feature_cols:
        # Global median on training set
        med_val = float(train_df[col].median())
        if np.isnan(med_val):
            med_val = 0.0  # Fallback if entirely null in train (edge case)
        global_medians[col] = med_val

        # Grouped medians by city
        if group_col and group_col in train_df.columns:
            grp = train_df.groupby(group_col)[col].median().to_dict()
            grouped_medians[col] = {str(k): float(v) for k, v in grp.items() if not np.isnan(v)}

    return {
        "feature_cols": feature_cols,
        "group_col": group_col,
        "global_medians": global_medians,
        "grouped_medians": grouped_medians
    }


def transform_impute(
    df: pd.DataFrame,
    imputer_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Apply pre-fitted imputation parameters to a dataset fold (train, val, or test).

    Args:
        df: Input DataFrame to impute.
        imputer_params: Parameters computed by fit_imputer().

    Returns:
        pd.DataFrame: Imputed DataFrame with zero missing values in feature_cols.
    """
    df_imputed = df.copy()
    feature_cols = imputer_params["feature_cols"]
    group_col = imputer_params.get("group_col")
    global_medians = imputer_params["global_medians"]
    grouped_medians = imputer_params.get("grouped_medians", {})

    for col in feature_cols:
        if col not in df_imputed.columns:
            continue

        missing_mask = df_imputed[col].isnull()
        if not missing_mask.any():
            continue

        # If grouped medians available, apply group-level median first
        if group_col and group_col in df_imputed.columns and col in grouped_medians:
            col_grp = grouped_medians[col]
            group_series = df_imputed[group_col].astype(str)
            mapped_medians = group_series.map(col_grp)
            df_imputed.loc[missing_mask, col] = mapped_medians[missing_mask]

        # Remaining nulls filled with global training median
        remaining_nulls = df_imputed[col].isnull()
        if remaining_nulls.any():
            fallback_val = global_medians.get(col, 0.0)
            df_imputed.loc[remaining_nulls, col] = fallback_val

    return df_imputed


def encode_target(df: pd.DataFrame, target_col: str = "AQI_Bucket") -> pd.DataFrame:
    """
    Encode the ordinal CPCB categories into integer labels (0 to 5).

    Mapping:
    - Good: 0
    - Satisfactory: 1
    - Moderate: 2
    - Poor: 3
    - Very Poor: 4
    - Severe: 5

    Adds a new column 'Target_Label' without removing the original text column.
    """
    df_out = df.copy()
    if target_col in df_out.columns:
        df_out["Target_Label"] = df_out[target_col].map(CATEGORY_TO_LABEL)
    return df_out


def save_processed_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Union[str, Path] = "data/processed"
) -> Dict[str, Path]:
    """
    Persist clean, preprocessed train/val/test splits to disk.
    Original raw files remain completely untouched.

    Returns:
        Dict[str, Path]: Dictionary mapping split names to saved file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    saved_files = {
        "train": out_path / "train.csv",
        "val": out_path / "val.csv",
        "test": out_path / "test.csv"
    }

    train_df.to_csv(saved_files["train"], index=False)
    val_df.to_csv(saved_files["val"], index=False)
    test_df.to_csv(saved_files["test"], index=False)

    return saved_files


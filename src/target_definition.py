"""
AirSense Target Definition & Leakage Prevention Module
======================================================
Defines, validates, and manages the ground-truth classification target for
the AirSense air-quality project.

Target:
- Primary Target: 'AQI_Bucket' (6-class ordinal categorical:
  'Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe').
- Standard & Provenance: Central Pollution Control Board (CPCB) of India
  National Air Quality Index (NAQI) Standard (2014/2015).

Key Safety & Anti-Leakage Constraints:
1. Target Integrity: Target labels are preserved from the documented dataset;
   rows with missing target are filtered rather than imputed.
2. Derivation Verification: Validates that AQI_Bucket strictly matches CPCB
   piecewise thresholds applied to continuous AQI.
3. Leakage Prevention: The continuous 'AQI' column is mathematically derived
   from pollutant sub-indices; it MUST be strictly excluded from input feature
   matrix X to prevent direct label leakage.
4. Task Demarcation: Predicts current-condition category from current pollutant
   concentrations. This diagnostic classification must never be conflated
   with future forecasting.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np

from src.data_loader import (
    POLLUTANT_COLUMNS,
    TARGET_COLUMNS,
    CPCB_AQI_CATEGORIES
)

# Documented CPCB NAQI Category Breakpoints
# CPCB NAQI Report (2014/2015), Ministry of Environment, Forest and Climate Change, India
CPCB_AQI_BREAKPOINTS: List[Tuple[float, float, str]] = [
    (0.0, 50.0, "Good"),
    (51.0, 100.0, "Satisfactory"),
    (101.0, 200.0, "Moderate"),
    (201.0, 300.0, "Poor"),
    (301.0, 400.0, "Very Poor"),
    (401.0, float("inf"), "Severe")
]

# Explicit columns that represent the target or direct target leakage
LEAKAGE_COLUMNS: List[str] = [
    "AQI",             # Continuous index from which AQI_Bucket is thresholded (Direct Leakage)
    "AQI_Bucket",      # Ground-truth label
    "Target_Label",    # Encoded numeric ground-truth label
    "Sub_Index_Max",   # Theoretical maximum sub-index (if present)
    "AQI_Calculated"   # Recomputed AQI (if present)
]


def aqi_to_cpcb_bucket(aqi_val: float) -> Optional[str]:
    """
    Derive the official CPCB AQI Category from a continuous numerical AQI value
    using the documented 2014 CPCB National Air Quality Index standard.

    CPCB Breakpoints:
    - 0 to 50: Good
    - 51 to 100: Satisfactory
    - 101 to 200: Moderate
    - 201 to 300: Poor
    - 301 to 400: Very Poor
    - 401+: Severe

    Args:
        aqi_val: Continuous numerical AQI scalar.

    Returns:
        str: Category name, or None if aqi_val is NaN or negative.
    """
    if pd.isna(aqi_val) or aqi_val < 0:
        return None

    for low, high, bucket in CPCB_AQI_BREAKPOINTS:
        if low <= aqi_val <= high:
            return bucket

    return "Severe" if aqi_val > 400 else None


def validate_target_labels(df: pd.DataFrame, target_col: str = "AQI_Bucket") -> Dict[str, Any]:
    """
    Audit the target column in the dataset:
    - Verifies label values against CPCB standards.
    - Counts missing, invalid, or formatted labels.
    - Confirms concordance between numerical 'AQI' and categorical 'AQI_Bucket'.

    Returns:
        Dict[str, Any]: Detailed audit summary.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    total_rows = len(df)
    target_series = df[target_col]
    null_count = int(target_series.isnull().sum())
    valid_count = total_rows - null_count

    valid_series = target_series.dropna()
    unique_labels = sorted(valid_series.unique().tolist())
    expected_set = set(CPCB_AQI_CATEGORIES)
    observed_set = set(unique_labels)

    invalid_labels = list(observed_set - expected_set)
    label_counts = target_series.value_counts(dropna=False).to_dict()
    label_counts = {str(k): int(v) for k, v in label_counts.items()}

    # Check alignment between AQI and AQI_Bucket if AQI is available
    mismatch_count = 0
    if "AQI" in df.columns:
        has_both = df.dropna(subset=["AQI", target_col])
        derived = has_both["AQI"].apply(aqi_to_cpcb_bucket)
        mismatch_count = int((derived != has_both[target_col]).sum())

    is_valid = (len(invalid_labels) == 0) and (valid_count > 0)

    return {
        "target_col": target_col,
        "total_rows": total_rows,
        "valid_rows": valid_count,
        "null_rows": null_count,
        "null_pct": round((null_count / total_rows) * 100, 2) if total_rows > 0 else 0.0,
        "unique_labels": unique_labels,
        "expected_labels": CPCB_AQI_CATEGORIES,
        "invalid_labels": invalid_labels,
        "label_counts": label_counts,
        "aqi_bucket_mismatches": mismatch_count,
        "is_valid_standard": is_valid
    }


def identify_leakage_features(feature_names: List[str]) -> List[str]:
    """
    Scan a list of candidate feature names and identify any that would introduce
    data leakage (features that encode the target or are derived from it).

    Args:
        feature_names: List of column names under consideration as model inputs.

    Returns:
        List[str]: List of detected leakage column names that MUST be excluded.
    """
    detected_leakage = []
    for f in feature_names:
        f_lower = f.lower().strip()
        # Direct match with known leakage columns
        if f in LEAKAGE_COLUMNS:
            detected_leakage.append(f)
        # Any variant of AQI or target
        elif f_lower in ["aqi", "aqi_bucket", "aqi_category", "target", "target_label"]:
            detected_leakage.append(f)
        elif "aqi" in f_lower and not any(p in f_lower for p in ["pm", "no", "so", "co", "o3"]):
            detected_leakage.append(f)

    return detected_leakage


def prepare_feature_target_matrices(
    df: pd.DataFrame,
    target_col: str = "AQI_Bucket",
    include_temporal: bool = True,
    exclude_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Safely extract feature matrix X and target vector y from a cleaned dataset.
    Guarantees that all target and leakage columns are strictly excluded.

    Args:
        df: Input clean DataFrame (rows with missing targets should be dropped).
        target_col: Target column name (default 'AQI_Bucket').
        include_temporal: Whether to include safe calendar features (Month, Season, etc.).
        exclude_columns: Optional list of additional columns to drop.

    Returns:
        Tuple[pd.DataFrame, pd.Series, List[str]]:
            - X: Clean feature DataFrame (zero leakage columns).
            - y: Target Series (categorical or encoded).
            - feature_names: List of final feature names in X.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    # Drop any remaining null targets
    clean_df = df.dropna(subset=[target_col]).copy()
    y = clean_df[target_col].copy()

    # Determine columns to drop
    cols_to_drop = set(LEAKAGE_COLUMNS)
    cols_to_drop.add(target_col)
    cols_to_drop.add("Date")  # Raw date timestamp is dropped in favor of engineered calendar features

    if exclude_columns:
        cols_to_drop.update(exclude_columns)

    candidate_features = [c for c in clean_df.columns if c not in cols_to_drop]

    # Verify zero leakage features remain
    leakage = identify_leakage_features(candidate_features)
    if leakage:
        raise ValueError(
            f"CRITICAL ERROR: Data leakage columns detected in candidate features: {leakage}. "
            f"These must be excluded before modeling."
        )

    X = clean_df[candidate_features].copy()
    feature_names = list(X.columns)

    return X, y, feature_names


"""
AirSense Feature Engineering Module
=====================================
Defines and validates the feature set used across all four AirSense models
(Decision Tree, Random Forest, SVM, XGBoost).

Design Rationale:
-----------------
Features are drawn strictly from the city_day.csv dataset columns that
have been validated in previous EDA and preprocessing stages. No external
or synthetic weather variables are introduced.

Feature Categories:
-------------------
1. Pollutant Concentrations (12 numerical features):
   PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
   These are the primary CPCB-criteria pollutants measured at monitoring stations.

2. Temporal Calendar Features (numerical + categorical):
   Year, Month, Day, DayOfWeek, IsWeekend (numerical)
   Season (categorical: Winter, Summer, Monsoon, Post-Monsoon)

3. City Identifier (EXCLUDED — see rationale below):
   'City' is present in city_day.csv but is deliberately excluded by default.
   Rationale for EXCLUSION:
   - Including City would allow the model to memorize city-specific pollution
     patterns rather than learning from pollutant concentrations directly.
   - Models trained with City would fail to generalize to new cities or
     to cities whose pollution profile changes significantly over time.
   - It encodes geographic identity, not a measurable air quality feature.
   - Implication: a model including City would report inflated accuracy
     on seen cities but degrade sharply on novel locations.
   The include_city_flag parameter allows experimental inclusion only.

Target:
-------
AQI_Bucket: 6-class ordinal label (Good → Satisfactory → Moderate →
            Poor → Very Poor → Severe). Encoded to integers 0–5.

Leakage Prevention:
-------------------
AQI (continuous index) is strictly excluded from X because AQI_Bucket is
a deterministic piecewise threshold function of AQI. Including AQI would
allow trivial 100% accuracy by reading off the category boundary directly.
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

from src.data_loader import POLLUTANT_COLUMNS, CPCB_AQI_CATEGORIES
from src.preprocessing import SEASON_MAP, CATEGORY_TO_LABEL, extract_temporal_features

# ─────────────────────────────────────────────────────────────────
# 1. Feature Definitions
# ─────────────────────────────────────────────────────────────────

#: 12 criteria-pollutant concentration columns (numerical, continuous)
NUMERICAL_POLLUTANT_FEATURES: List[str] = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3", "Benzene",
    "Toluene", "Xylene"
]

#: Safe temporal features derived deterministically from the Date column
NUMERICAL_TEMPORAL_FEATURES: List[str] = [
    "Year", "Month", "Day", "DayOfWeek", "IsWeekend"
]

#: Categorical temporal features requiring encoding
CATEGORICAL_TEMPORAL_FEATURES: List[str] = ["Season"]

#: All numerical features (pollutants + temporal numerics)
ALL_NUMERICAL_FEATURES: List[str] = NUMERICAL_POLLUTANT_FEATURES + NUMERICAL_TEMPORAL_FEATURES

#: All categorical features (only Season by default; City is opt-in)
ALL_CATEGORICAL_FEATURES: List[str] = CATEGORICAL_TEMPORAL_FEATURES

#: Default features used for modeling (no City)
DEFAULT_FEATURES: List[str] = ALL_NUMERICAL_FEATURES + ALL_CATEGORICAL_FEATURES

#: Target column name
TARGET_COL: str = "AQI_Bucket"

#: Encoded target column name
TARGET_LABEL_COL: str = "Target_Label"

#: Columns strictly forbidden in X (leakage risk)
LEAKAGE_COLS: List[str] = ["AQI", "AQI_Bucket", "Target_Label", "AQI_Calculated", "Sub_Index_Max"]

#: Columns to always drop before feature matrix construction
ALWAYS_DROP: List[str] = LEAKAGE_COLS + ["Date", "City"]


# ─────────────────────────────────────────────────────────────────
# 2. Dataset Preparation Helper
# ─────────────────────────────────────────────────────────────────

def prepare_modeling_dataset(
    df: pd.DataFrame,
    include_city: bool = False
) -> pd.DataFrame:
    """
    Prepare a raw (or partially cleaned) DataFrame for modeling by:
    1. Extracting temporal features from 'Date' column.
    2. Encoding the target label to integers.

    Args:
        df: DataFrame loaded from city_day.csv or a preprocessed version.
        include_city: Whether to retain the 'City' column as a feature.
                      Default is False (excluded for generalizability).

    Returns:
        pd.DataFrame: Enhanced DataFrame with temporal features and Target_Label.
    """
    df_out = df.copy()

    # Extract calendar features from Date column if present
    if "Date" in df_out.columns:
        df_out = extract_temporal_features(df_out, date_col="Date")

    # Encode target label
    if TARGET_COL in df_out.columns:
        df_out[TARGET_LABEL_COL] = df_out[TARGET_COL].map(CATEGORY_TO_LABEL)

    return df_out


def get_feature_lists(
    df: pd.DataFrame,
    include_city: bool = False
) -> Dict[str, List[str]]:
    """
    Inspect the available columns in a prepared DataFrame and return
    explicit lists of numerical and categorical features available for modeling.

    This function filters to only columns that actually exist in df,
    so it is safe to call on datasets where some features may be absent.

    Args:
        df: A DataFrame that has gone through prepare_modeling_dataset().
        include_city: Whether to include 'City' as a categorical predictor.

    Returns:
        Dict with keys:
            'numerical'   → list of numerical feature column names
            'categorical' → list of categorical feature column names
            'all'         → combined ordered list (numerical first)
    """
    available = set(df.columns)

    numerical = [f for f in ALL_NUMERICAL_FEATURES if f in available]
    categorical = [f for f in ALL_CATEGORICAL_FEATURES if f in available]

    if include_city and "City" in available:
        categorical = categorical + ["City"]

    return {
        "numerical": numerical,
        "categorical": categorical,
        "all": numerical + categorical
    }


def extract_X_y(
    df: pd.DataFrame,
    feature_lists: Optional[Dict[str, List[str]]] = None,
    use_integer_labels: bool = True,
    include_city: bool = False
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract the feature matrix X and target vector y from a prepared DataFrame.

    Performs a final leakage check — raises ValueError if any leakage
    column ends up in X.

    Args:
        df: DataFrame processed by prepare_modeling_dataset().
        feature_lists: Optional dict from get_feature_lists(). If None,
                       it is computed automatically.
        use_integer_labels: If True, y contains integer labels (0–5).
                            If False, y contains string category names.
        include_city: Forwarded to get_feature_lists() if feature_lists is None.

    Returns:
        Tuple[pd.DataFrame, pd.Series]:
            X — feature matrix with no leakage columns
            y — target vector
    """
    if feature_lists is None:
        feature_lists = get_feature_lists(df, include_city=include_city)

    all_features = feature_lists["all"]

    # Leakage guard: ensure no forbidden column is in the feature list
    detected_leakage = [col for col in all_features if col in LEAKAGE_COLS]
    if detected_leakage:
        raise ValueError(
            f"Data leakage detected! The following columns must not appear in X: "
            f"{detected_leakage}. Remove them from feature_lists before proceeding."
        )

    # Only keep features that exist in df
    available_features = [f for f in all_features if f in df.columns]

    # Extract X
    X = df[available_features].copy()

    # Extract y
    if use_integer_labels:
        if TARGET_LABEL_COL not in df.columns:
            raise KeyError(
                f"'{TARGET_LABEL_COL}' not found in DataFrame. "
                "Run prepare_modeling_dataset() first."
            )
        y = df[TARGET_LABEL_COL].copy()
    else:
        if TARGET_COL not in df.columns:
            raise KeyError(f"'{TARGET_COL}' not found in DataFrame.")
        y = df[TARGET_COL].copy()

    # Drop rows where y is null (target integrity rule)
    valid_mask = y.notna()
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)

    return X, y


def summarize_features(df: pd.DataFrame, feature_lists: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Generate a summary table of feature statistics for documentation purposes.

    Returns:
        pd.DataFrame: Summary with columns for feature name, type, missing count/pct,
                      min, max, and unique values.
    """
    records = []
    for feat in feature_lists["numerical"]:
        if feat not in df.columns:
            continue
        col = df[feat]
        records.append({
            "Feature": feat,
            "Type": "Numerical",
            "Missing": int(col.isnull().sum()),
            "Missing_Pct": round(col.isnull().mean() * 100, 2),
            "Min": round(float(col.min()), 4) if col.notna().any() else None,
            "Max": round(float(col.max()), 4) if col.notna().any() else None,
            "Unique": int(col.nunique()),
        })
    for feat in feature_lists["categorical"]:
        if feat not in df.columns:
            continue
        col = df[feat]
        records.append({
            "Feature": feat,
            "Type": "Categorical",
            "Missing": int(col.isnull().sum()),
            "Missing_Pct": round(col.isnull().mean() * 100, 2),
            "Min": None,
            "Max": None,
            "Unique": int(col.nunique()),
        })
    return pd.DataFrame(records)

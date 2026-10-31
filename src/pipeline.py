"""
AirSense Preprocessing Pipeline Module
========================================
Implements a fully scikit-learn-compatible, reusable preprocessing pipeline
for the AirSense air quality classification benchmark.

Architecture Overview:
----------------------
This module wraps all data-preparation steps into a single scikit-learn
Pipeline + ColumnTransformer composition that:

  1. Handles missing values in numerical features (median imputation).
  2. Scales numerical features (StandardScaler — required for SVM, harmless for trees).
  3. Encodes the 'Season' categorical feature using OrdinalEncoder with
     safe handling for previously unseen category values.
  4. Produces a consistently ordered, fixed-width numeric array suitable
     for all four models: Decision Tree, Random Forest, SVM, XGBoost.

Key Design Principles:
-----------------------
- Fit on training data only: fit() / fit_transform() is called ONLY on
  training X. The validation and test splits are transform()-only.
  This is enforced by the caller, not enforced internally (by design of
  scikit-learn Pipeline).

- No target leakage: Imputation uses pollutant-median statistics computed
  from training X only. The target column y is never used to compute
  any imputer or scaler statistic.

- Unseen category safety: OrdinalEncoder is configured with
  handle_unknown='use_encoded_value' and unknown_value=-1, so a city
  or season value never seen in training receives a -1 placeholder
  without raising an error at inference time.

- Feature order consistency: ColumnTransformer preserves a fixed output
  column ordering across train/val/test. The utility function
  `get_feature_names_out()` returns the ordered list post-transformation.

- Scaling scope: StandardScaler is applied to ALL numerical features.
  For Decision Trees and Random Forest, scaling does not affect split
  decisions (trees are invariant to monotone transformations), but it
  does NOT hurt and ensures the same pipeline works for SVM without
  modification.

- No target imputation: Missing target rows must be removed before
  passing data to this pipeline. This is enforced in the data loading
  stage (filter_labeled_data / stratified_data_split).

Usage Example:
--------------
    from src.pipeline import build_preprocessing_pipeline, fit_pipeline
    from src.feature_engineering import get_feature_lists, extract_X_y, prepare_modeling_dataset

    # Prepare datasets
    train_prep = prepare_modeling_dataset(train_df)
    val_prep   = prepare_modeling_dataset(val_df)
    test_prep  = prepare_modeling_dataset(test_df)

    # Identify feature roles
    feature_lists = get_feature_lists(train_prep)

    # Extract X and y for each split
    X_train, y_train = extract_X_y(train_prep, feature_lists)
    X_val,   y_val   = extract_X_y(val_prep,   feature_lists)
    X_test,  y_test  = extract_X_y(test_prep,  feature_lists)

    # Build and fit pipeline (on training data ONLY)
    pipeline = build_preprocessing_pipeline(feature_lists)
    X_train_t = pipeline.fit_transform(X_train)
    X_val_t   = pipeline.transform(X_val)
    X_test_t  = pipeline.transform(X_test)
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from src.feature_engineering import (
    ALL_NUMERICAL_FEATURES,
    ALL_CATEGORICAL_FEATURES,
    CATEGORICAL_TEMPORAL_FEATURES,
    TARGET_COL,
    TARGET_LABEL_COL,
    LEAKAGE_COLS,
    prepare_modeling_dataset,
    get_feature_lists,
    extract_X_y,
)

# Known Season categories in the correct ordinal order (coldest → hottest → wet → post-wet)
SEASON_CATEGORIES: List[str] = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]


# ─────────────────────────────────────────────────────────────────
# Custom Transformer: Clip extreme values before scaling
# ─────────────────────────────────────────────────────────────────

class WinsorizeTransformer(BaseEstimator, TransformerMixin):
    """
    Clip numerical features at training-set percentile bounds to reduce the
    influence of sensor outliers.

    Why:
        Monitoring stations occasionally report implausibly large readings
        (instrument spikes, calibration errors). Clipping at the 1st and 99th
        percentiles of the training distribution bounds these values without
        discarding the row entirely.

    This transformer is fitted on training data (computing lo/hi bounds)
    and applied uniformly to all splits.
    """

    def __init__(self, lower_pct: float = 1.0, upper_pct: float = 99.0):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct
        self.lo_: Optional[np.ndarray] = None
        self.hi_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        self.lo_ = np.nanpercentile(X_arr, self.lower_pct, axis=0)
        self.hi_ = np.nanpercentile(X_arr, self.upper_pct, axis=0)
        return self

    def transform(self, X, y=None):
        X_arr = np.asarray(X, dtype=float).copy()
        X_arr = np.clip(X_arr, self.lo_, self.hi_)
        return X_arr

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features, dtype=object)
        return np.array([], dtype=object)


# ─────────────────────────────────────────────────────────────────
# Pipeline Builder
# ─────────────────────────────────────────────────────────────────

def build_preprocessing_pipeline(
    feature_lists: Dict[str, List[str]],
    scale_numerical: bool = True,
    winsorize: bool = True,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0
) -> Pipeline:
    """
    Construct a scikit-learn Pipeline that preprocesses the AirSense feature matrix.

    Pipeline stages for numerical columns:
        1. WinsorizeTransformer — clip outliers at training percentile bounds.
        2. SimpleImputer(strategy='median') — fill remaining NaN values with
           the column median computed from training data only.
        3. StandardScaler — zero-mean, unit-variance normalization.
           Required for SVM; monotone invariance makes it harmless for trees.

    Pipeline stages for categorical columns (Season, optionally City):
        1. SimpleImputer(strategy='most_frequent') — fill any rare NaN values.
        2. OrdinalEncoder — maps each category string to an integer.
           Unknown categories at inference time receive -1 (no exception raised).
           For Season, the known categories are explicitly provided to ensure
           consistent encoding across train/val/test regardless of which
           seasons appear in each split.

    Args:
        feature_lists: Dict from get_feature_lists() with keys
                       'numerical' and 'categorical'.
        scale_numerical: Whether to apply StandardScaler to numerical features.
                         Set False only for debugging/investigation.
        winsorize: Whether to apply WinsorizeTransformer before imputation.
        lower_pct: Lower clipping percentile for WinsorizeTransformer.
        upper_pct: Upper clipping percentile for WinsorizeTransformer.

    Returns:
        sklearn.pipeline.Pipeline: Unfitted preprocessing pipeline ready for
        fit_transform(X_train) and transform(X_val/X_test).
    """
    numerical_cols = feature_lists["numerical"]
    categorical_cols = feature_lists["categorical"]

    # ── Numerical sub-pipeline ──────────────────────────────────────
    num_steps = []
    if winsorize:
        num_steps.append(("winsorize", WinsorizeTransformer(lower_pct, upper_pct)))
    num_steps.append(("imputer", SimpleImputer(strategy="median")))
    if scale_numerical:
        num_steps.append(("scaler", StandardScaler()))
    num_pipeline = Pipeline(steps=num_steps)

    # ── Categorical sub-pipeline ────────────────────────────────────
    # Build per-column known-categories list for OrdinalEncoder
    cat_categories = []
    for col in categorical_cols:
        if col == "Season":
            cat_categories.append(SEASON_CATEGORIES)
        else:
            # For 'City' or any other categorical: 'auto' means learn from training data
            cat_categories.append("auto")

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=cat_categories if cat_categories else "auto",
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float64
        ))
    ])

    # ── ColumnTransformer ──────────────────────────────────────────
    transformers = []
    if numerical_cols:
        transformers.append(("numerical", num_pipeline, numerical_cols))
    if categorical_cols:
        transformers.append(("categorical", cat_pipeline, categorical_cols))

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",      # Explicitly drop any unspecified columns
        verbose_feature_names_out=True  # Prefix feature names for clarity
    )

    # ── Final pipeline (just the preprocessor; model is added separately) ──
    preprocessing_pipeline = Pipeline(steps=[
        ("preprocessor", column_transformer)
    ])

    return preprocessing_pipeline


# ─────────────────────────────────────────────────────────────────
# Convenience: Fit Pipeline on Training Data
# ─────────────────────────────────────────────────────────────────

def fit_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: Optional[pd.Series] = None
) -> Pipeline:
    """
    Fit the preprocessing pipeline on training data only.

    Args:
        pipeline: Unfitted pipeline from build_preprocessing_pipeline().
        X_train: Training feature matrix (DataFrame).
        y_train: Not used in preprocessing, included for API compatibility.

    Returns:
        Pipeline: The same pipeline object, now fitted on training data.
    """
    pipeline.fit(X_train)
    return pipeline


def transform_split(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Apply a fitted preprocessing pipeline to a dataset split (val or test).

    Args:
        pipeline: Fitted pipeline (must have been fitted on training data).
        X: Feature DataFrame (val or test).

    Returns:
        np.ndarray: Transformed feature array.
    """
    return pipeline.transform(X)


def get_feature_names_out(
    pipeline: Pipeline,
    feature_lists: Dict[str, List[str]]
) -> List[str]:
    """
    Return the ordered list of output feature names after transformation.

    ColumnTransformer with verbose_feature_names_out=True prefixes each
    feature name with its transformer name (e.g., 'numerical__PM2.5').

    Args:
        pipeline: Fitted preprocessing pipeline.
        feature_lists: Dict from get_feature_lists().

    Returns:
        List[str]: Ordered feature names corresponding to transformed columns.
    """
    try:
        ct = pipeline.named_steps["preprocessor"]
        names = list(ct.get_feature_names_out())
        return names
    except Exception:
        # Fallback: construct names manually
        numerical = [f"numerical__{f}" for f in feature_lists["numerical"]]
        categorical = [f"categorical__{f}" for f in feature_lists["categorical"]]
        return numerical + categorical


# ─────────────────────────────────────────────────────────────────
# End-to-End Data Preparation Helper
# ─────────────────────────────────────────────────────────────────

def prepare_all_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    include_city: bool = False,
    scale_numerical: bool = True,
    winsorize: bool = True
) -> Dict[str, Any]:
    """
    Full end-to-end preparation of train/val/test splits.

    Steps:
    1. Extract temporal features from Date column.
    2. Determine feature lists (numerical + categorical).
    3. Extract X and y for each split.
    4. Build and FIT preprocessing pipeline on training data only.
    5. Transform all three splits using the fitted pipeline.

    Args:
        train_df: Raw training DataFrame (with Date, AQI_Bucket, pollutants).
        val_df: Raw validation DataFrame.
        test_df: Raw test DataFrame.
        include_city: Whether to include City as a categorical predictor.
        scale_numerical: Whether to apply StandardScaler.
        winsorize: Whether to clip outliers at percentile bounds.

    Returns:
        Dict with keys:
            'X_train_t', 'X_val_t', 'X_test_t'   — transformed arrays
            'y_train', 'y_val', 'y_test'           — integer-encoded targets
            'pipeline'                              — fitted preprocessing pipeline
            'feature_lists'                        — dict of numerical/categorical features
            'feature_names_out'                    — ordered output feature names
            'label_map'                            — int → category string mapping
    """
    from src.preprocessing import LABEL_TO_CATEGORY

    # Step 1: Add temporal features + encode target label
    train_prep = prepare_modeling_dataset(train_df, include_city=include_city)
    val_prep   = prepare_modeling_dataset(val_df,   include_city=include_city)
    test_prep  = prepare_modeling_dataset(test_df,  include_city=include_city)

    # Step 2: Identify feature roles from training data only
    feature_lists = get_feature_lists(train_prep, include_city=include_city)

    # Step 3: Extract X, y for each split
    X_train, y_train = extract_X_y(train_prep, feature_lists)
    X_val,   y_val   = extract_X_y(val_prep,   feature_lists)
    X_test,  y_test  = extract_X_y(test_prep,  feature_lists)

    # Step 4: Build and fit pipeline on training data ONLY
    pipeline = build_preprocessing_pipeline(
        feature_lists,
        scale_numerical=scale_numerical,
        winsorize=winsorize
    )
    X_train_t = pipeline.fit_transform(X_train)

    # Step 5: Transform val/test using the fitted pipeline
    X_val_t  = pipeline.transform(X_val)
    X_test_t = pipeline.transform(X_test)

    feature_names_out = get_feature_names_out(pipeline, feature_lists)

    return {
        "X_train_t": X_train_t,
        "X_val_t":   X_val_t,
        "X_test_t":  X_test_t,
        "y_train":   y_train,
        "y_val":     y_val,
        "y_test":    y_test,
        "pipeline":  pipeline,
        "feature_lists":      feature_lists,
        "feature_names_out":  feature_names_out,
        "label_map":          LABEL_TO_CATEGORY,
        "X_train_df": X_train,
        "X_val_df":   X_val,
        "X_test_df":  X_test,
    }

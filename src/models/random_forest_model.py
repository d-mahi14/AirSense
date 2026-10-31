"""
AirSense Random Forest Classifier Module
==========================================
Implements Random Forest classification for AirSense air quality prediction.
Uses sklearn RandomForestClassifier integrated with the AirSense preprocessing
pipeline, reusing the exact same train/val/test partitions as the Decision Tree.

What is Ensemble Learning?
----------------------------
Instead of training a single model, ensemble methods train many models
and aggregate their predictions. The intuition: a crowd of mediocre
independent forecasters often beats a single "expert" forecaster.

Two main families:
  - Bagging: Train models on bootstrap samples; average predictions.
  - Boosting: Train models sequentially, each correcting the prior's errors.

Random Forest is a bagging ensemble of Decision Trees.

Bootstrap Sampling:
-------------------
Given a training set of N samples, each tree in the forest is trained
on a bootstrap sample — a random draw of N samples WITH replacement.
On average, ~63.2% of training rows appear in each bootstrap sample
(the rest are "out-of-bag" rows, useful for internal validation).
Bootstrap ensures each tree sees a slightly different training set,
so the trees make different errors.

Random Feature Selection:
--------------------------
At each node split, Random Forest does NOT consider all P features.
Instead, it randomly selects a subset of √P (for classification) features
and finds the best split among only those candidates.

Why restrict features? If one very strong predictor exists (e.g., PM2.5),
every tree in a plain bagging ensemble would use it for the first split,
making all trees highly correlated. Correlated errors do not average out.
Random feature selection de-correlates the trees, making the forest
more robust.

Majority Voting:
-----------------
For classification, each of the K trees makes a class prediction.
The final prediction is the majority vote across all K trees:
    ŷ = argmax_c Σ_{k=1}^{K} 1[tree_k predicts c]

Single Tree vs. Forest:
------------------------
| Property         | Single Decision Tree | Random Forest       |
|------------------|---------------------|---------------------|
| Variance         | High                | Low                 |
| Bias             | Low                 | Slightly higher     |
| Interpretability | Full tree viewable  | Black-box           |
| Training time    | Fast                | K × tree time       |
| Overfitting risk | High                | Much lower          |
| Feature import.  | Single tree         | Averaged importance |

Advantages for AirSense:
-------------------------
- Robustness to outliers: Averaging across hundreds of trees dampens
  the effect of anomalous sensor readings.
- Built-in feature importance: Tracks how much each pollutant contributes
  to classification across all trees.
- Handles missing value imputation well: With median-imputed features,
  forests are resilient to imputation noise.
- Excellent out-of-the-box performance: Often requires minimal tuning.

Limitations:
------------
- Slower than single trees (must train K trees).
- Less interpretable: Cannot visualize the full ensemble.
- Memory usage scales with K × tree size.
- Like Decision Trees, constrained to axis-aligned boundaries.
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from src.preprocessing import LABEL_TO_CATEGORY, CATEGORY_TO_LABEL
from src.data_loader import CPCB_AQI_CATEGORIES
from src.models.decision_tree_model import (
    evaluate_model,
    save_model_pipeline,
    save_model_config,
    print_evaluation_report
)

DEFAULT_MODEL_DIR = Path("models")


def build_random_forest_pipeline(
    preprocessing_pipeline: Pipeline,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    max_features: str = "sqrt",
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
    n_jobs: int = -1
) -> Pipeline:
    """
    Build a Pipeline combining the AirSense preprocessor with RandomForestClassifier.

    Why these defaults?
    -------------------
    - n_estimators=200: 200 trees provides stable variance reduction without
      excessive training time. More trees are always better but with diminishing
      returns past ~100–300.
    - max_depth=None: Trees are grown fully (each leaf has min_samples_leaf
      samples). Forest averaging prevents overfitting better than depth limits.
    - min_samples_split=10: Prevents splitting very small groups.
    - min_samples_leaf=5: Ensures each leaf represents meaningful data.
    - max_features='sqrt': Standard for classification — at each split,
      consider √(n_features) randomly selected features.
    - class_weight='balanced': Compensates for CPCB class imbalance
      (Good=5.4%, Moderate=35.5%). Each class gets weight = N / (n_classes × n_k).
    - n_jobs=-1: Parallelise across all available CPU cores.

    Args:
        preprocessing_pipeline: Fitted or unfitted preprocessing Pipeline.
        n_estimators: Number of trees to grow.
        max_depth: Maximum depth per tree (None = unlimited).
        min_samples_split: Minimum samples to consider a split.
        min_samples_leaf: Minimum samples per leaf node.
        max_features: Feature subset size at each split.
        class_weight: Class weighting strategy.
        random_state: Reproducibility seed.
        n_jobs: CPU parallelism (-1 = all cores).

    Returns:
        sklearn.pipeline.Pipeline: Full Pipeline (preprocessing + forest).
    """
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        oob_score=True  # Use out-of-bag samples for internal validation
    )

    full_pipeline = Pipeline(
        steps=preprocessing_pipeline.steps + [("classifier", classifier)]
    )
    return full_pipeline


def get_feature_importances(
    pipeline: Pipeline,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extract and rank feature importances from the trained RandomForest.

    Feature importance in a Random Forest is computed as the mean decrease
    in node impurity (MDI) attributable to each feature, averaged across
    all trees. Higher values indicate features more frequently selected for
    high-gain splits.

    Args:
        pipeline: Fitted pipeline containing a RandomForestClassifier.
        feature_names: Ordered list of feature names matching pipeline output.

    Returns:
        pd.DataFrame: Features ranked by importance (descending).
    """
    rf_model = pipeline.named_steps["classifier"]
    importances = rf_model.feature_importances_

    n_feats = min(len(feature_names), len(importances))
    df_imp = pd.DataFrame({
        "Feature": feature_names[:n_feats],
        "Importance": importances[:n_feats]
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return df_imp


def run_random_forest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    hyperparams: Optional[Dict[str, Any]] = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
    include_city: bool = False
) -> Dict[str, Any]:
    """
    Full end-to-end Random Forest training and evaluation run.

    Uses the same data partitions and preprocessing as the Decision Tree model.

    Args:
        train_df: Raw training DataFrame.
        val_df: Raw validation DataFrame.
        test_df: Raw test DataFrame.
        hyperparams: Optional hyperparameter overrides.
        model_dir: Directory to save model and config.
        include_city: Whether to include City as a categorical feature.

    Returns:
        Dict[str, Any]: Results including pipeline, metrics, and feature importances.
    """
    from src.pipeline import prepare_all_splits

    # Default hyperparameters
    hp = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    if hyperparams:
        hp.update(hyperparams)

    print("  Preparing data splits...")
    data = prepare_all_splits(
        train_df, val_df, test_df,
        include_city=include_city,
        scale_numerical=True,
        winsorize=True
    )

    preproc_pipeline = data["pipeline"]

    print("  Building Random Forest pipeline...")
    full_pipeline = build_random_forest_pipeline(preproc_pipeline, **hp)

    X_train_df = data["X_train_df"]
    X_val_df   = data["X_val_df"]
    X_test_df  = data["X_test_df"]
    y_train    = data["y_train"]
    y_val      = data["y_val"]
    y_test     = data["y_test"]

    print(f"  Training Random Forest ({hp['n_estimators']} trees)...")
    t_start = time.perf_counter()
    full_pipeline.fit(X_train_df, y_train)
    train_time = round(time.perf_counter() - t_start, 4)
    print(f"  Training complete in {train_time:.2f}s.")

    # Out-of-bag score (forest's internal validation)
    rf_model = full_pipeline.named_steps["classifier"]
    oob = round(rf_model.oob_score_, 4) if hasattr(rf_model, "oob_score_") else None
    if oob is not None:
        print(f"  Out-of-Bag (OOB) Score: {oob:.4f}")

    # Evaluate
    train_metrics = evaluate_model(full_pipeline, X_train_df, y_train, "Train")
    val_metrics   = evaluate_model(full_pipeline, X_val_df,   y_val,   "Validation")
    test_metrics  = evaluate_model(full_pipeline, X_test_df,  y_test,  "Test")

    print_evaluation_report(train_metrics)
    print_evaluation_report(val_metrics)
    print_evaluation_report(test_metrics)

    # Feature importances
    feat_importances = get_feature_importances(
        full_pipeline, data["feature_names_out"]
    )
    print("\n  Top 10 Feature Importances:")
    print(feat_importances.head(10).to_string(index=False))

    # Save model
    save_path   = save_model_pipeline(full_pipeline, "random_forest", model_dir)
    config = {
        "model": "RandomForestClassifier",
        "hyperparameters": {k: str(v) if v is None else v for k, v in hp.items()},
        "feature_lists": data["feature_lists"],
        "feature_names_out": data["feature_names_out"],
        "train_time_sec": train_time,
        "oob_score": oob,
        "n_train": len(y_train),
        "n_val":   len(y_val),
        "n_test":  len(y_test),
        "train_accuracy": train_metrics["accuracy"],
        "val_accuracy":   val_metrics["accuracy"],
        "test_accuracy":  test_metrics["accuracy"],
        "val_macro_f1":   val_metrics["macro_f1"],
        "test_macro_f1":  test_metrics["macro_f1"],
    }
    config_path = save_model_config(config, "random_forest", model_dir)

    print(f"\n  Model saved to: {save_path}")
    print(f"  Config saved to: {config_path}")

    return {
        "pipeline": full_pipeline,
        "train_metrics": train_metrics,
        "val_metrics":   val_metrics,
        "test_metrics":  test_metrics,
        "train_time_sec": train_time,
        "oob_score": oob,
        "feature_importances": feat_importances,
        "config": config,
        "data": data,
    }


if __name__ == "__main__":
    from src.data_loader import load_city_day
    from src.preprocessing import filter_labeled_data
    from src.data_split import stratified_data_split

    print("Loading dataset...")
    df = load_city_day()
    df, audit = filter_labeled_data(df)
    print(f"Labeled rows: {audit['retained_rows']:,}  (dropped: {audit['dropped_missing_target']:,})")

    train_df, val_df, test_df = stratified_data_split(df, random_state=42)
    print(f"Split sizes — Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    results = run_random_forest(train_df, val_df, test_df)

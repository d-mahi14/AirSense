"""
AirSense XGBoost Classifier Module
=====================================
Implements XGBoost classification for AirSense air quality prediction.
Uses XGBClassifier (xgboost library) with correct multiclass configuration
integrated into the AirSense preprocessing pipeline.

What is Gradient Boosting?
---------------------------
Gradient Boosting is a sequential ensemble method that builds an additive
model of "weak learners" (typically shallow Decision Trees), where each
new tree corrects the residual errors of all previous trees.

The key insight: instead of fitting a new model to the data, fit a new
model to the negative gradient of the loss function — the direction in
which the combined predictions need to move to reduce errors.

Sequential Construction of Trees:
----------------------------------
Given a loss function L(y, F(x)), the algorithm builds the model F(x):

  Step 0: Initialize F₀(x) = argmin_γ Σ L(yᵢ, γ)  [constant prediction]

  For m = 1, 2, ..., M:
    Step 1: Compute pseudo-residuals (negative gradients):
            rᵢₘ = −[∂L(yᵢ, F(xᵢ))/∂F(xᵢ)]  for all i

    Step 2: Fit a decision tree hₘ(x) to {(xᵢ, rᵢₘ)}

    Step 3: Update predictions:
            F_m(x) = F_{m-1}(x) + η × hₘ(x)
            where η = learning_rate (shrinkage)

For multiclass AQI (6 classes), XGBoost uses the softmax loss function:
    L(y, F) = −Σₖ yₖ log(pₖ)    where  pₖ = softmax(Fₖ)

One tree-per-class is built per boosting round:
    6 classes × M rounds = 6M trees total.

Loss Functions and Gradient Information:
-----------------------------------------
At each round, the gradient tells each tree: "For each training example,
how much and in which direction should the prediction move?"

For AirSense, when the model confuses 'Moderate' with 'Poor', the
gradient signal pushes the next tree to focus on those boundary cases.

Regularization in XGBoost:
---------------------------
XGBoost extends standard gradient boosting with L1/L2 regularization:
  Ω(tree) = γ × T + (λ/2) × Σⱼ wⱼ²

where T = number of leaves, wⱼ = leaf weight, γ = leaf-count penalty,
λ = L2 regularization on leaf weights.

This explicit regularization reduces overfitting beyond just limiting depth.

Learning Rate and Boosting Rounds:
------------------------------------
The learning rate η (also called "shrinkage"):
- Each tree's contribution is multiplied by η before adding it to F.
- Small η (e.g., 0.05) → each tree makes a tiny step → more trees needed
  for convergence → but final model is more robust.
- Large η (e.g., 0.3) → fast learning → fewer trees → risk of overfitting.

Optimal practice: use small η (0.05–0.1) with early stopping to find the
optimal number of rounds M automatically on a validation set.

Subsample and Colsample (Stochastic Gradient Boosting):
---------------------------------------------------------
- subsample: Fraction of training rows used per boosting round (row sampling).
  E.g., subsample=0.8 → each tree sees a random 80% of training rows.
  Reduces overfitting, speeds training.
- colsample_bytree: Fraction of features sampled per tree.
  Similar to Random Forest's max_features — de-correlates trees.

XGBoost vs. Random Forest:
----------------------------
| Property           | Random Forest        | XGBoost              |
|--------------------|----------------------|----------------------|
| Learning strategy  | Bagging (parallel)   | Boosting (sequential)|
| Tree correlation   | De-correlated        | Intentionally correlated|
| Fitting target     | Original labels      | Residuals/gradients  |
| Overfitting risk   | Lower               | Higher without tuning|
| Speed (train)      | Parallelizable       | Sequential, fast API |
| Regularization     | Implicit (bootstrap) | Explicit (λ, γ)     |
| Performance        | Strong baseline      | Often state-of-art   |

Handling Class Labels:
-----------------------
XGBoost's multiclass objective ('multi:softmax' or 'multi:softprob')
requires integer labels in [0, n_classes−1]. The AirSense pipeline
provides Target_Label ∈ {0,1,2,3,4,5} which maps directly.
"""

import time
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib

try:
    from xgboost import XGBClassifier
    import xgboost as xgb
    XGBOOST_VERSION = xgb.__version__
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBOOST_VERSION = None

from sklearn.pipeline import Pipeline

from src.preprocessing import LABEL_TO_CATEGORY
from src.data_loader import CPCB_AQI_CATEGORIES
from src.models.decision_tree_model import (
    evaluate_model,
    save_model_pipeline,
    save_model_config,
    print_evaluation_report
)

DEFAULT_MODEL_DIR = Path("models")


def check_xgboost() -> str:
    """
    Verify XGBoost is installed and return its version string.

    Raises:
        ImportError: If xgboost is not installed, with installation instructions.
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError(
            "XGBoost is not installed. Install it with:\n"
            "  pip install xgboost\n"
            "Or in PowerShell:\n"
            "  python.exe -m pip install xgboost"
        )
    return XGBOOST_VERSION


def build_xgboost_pipeline(
    preprocessing_pipeline: Pipeline,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    scale_pos_weight: Optional[float] = None,
    random_state: int = 42,
    n_jobs: int = -1,
    eval_metric: str = "mlogloss",
    n_classes: int = 6
) -> Pipeline:
    """
    Build a Pipeline combining the AirSense preprocessor with XGBClassifier.

    Why these defaults?
    -------------------
    - n_estimators=300: 300 boosting rounds with small learning_rate=0.05
      gives a thorough search through the loss landscape. Early stopping
      (applied separately with a validation set) finds the true optimal M.
    - max_depth=6: Moderately deep trees. XGBoost's regularization (λ, α)
      prevents overfitting at this depth. Typically 4–8 for tabular data.
    - learning_rate=0.05: Small shrinkage → more trees needed but more robust
      generalisation. Standard starting point.
    - subsample=0.8: Each tree sees 80% of training rows → stochastic gradient
      boosting, reduces overfitting.
    - colsample_bytree=0.8: Each tree selects 80% of features → de-correlates
      trees, speeds training.
    - reg_alpha=0.1, reg_lambda=1.0: Mild L1+L2 regularization on leaf weights.
    - eval_metric='mlogloss': Multiclass log-loss — appropriate for 6-class
      classification.
    - n_classes=6: Explicitly set for the 6 CPCB categories.

    Args:
        preprocessing_pipeline: AirSense preprocessing Pipeline.
        n_estimators: Maximum number of boosting rounds.
        max_depth: Maximum depth of each tree.
        learning_rate: Shrinkage factor η.
        subsample: Row subsampling ratio per tree.
        colsample_bytree: Column subsampling ratio per tree.
        reg_alpha: L1 regularization on leaf weights.
        reg_lambda: L2 regularization on leaf weights.
        scale_pos_weight: Weight for positive class (binary use only; None for multiclass).
        random_state: Reproducibility seed.
        n_jobs: Parallelism (-1 = all cores).
        eval_metric: Evaluation metric for early stopping.
        n_classes: Number of target classes.

    Returns:
        sklearn.pipeline.Pipeline: Full Pipeline (preprocessing + XGBClassifier).
    """
    check_xgboost()

    classifier = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective="multi:softmax",
        num_class=n_classes,
        eval_metric=eval_metric,
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=0,
        use_label_encoder=False if hasattr(XGBClassifier, 'use_label_encoder') else None
    )

    # Remove None-valued kwargs (older XGBoost versions may not accept some params)
    full_pipeline = Pipeline(
        steps=preprocessing_pipeline.steps + [("classifier", classifier)]
    )
    return full_pipeline


def run_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    hyperparams: Optional[Dict[str, Any]] = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
    include_city: bool = False,
    use_early_stopping: bool = True,
    early_stopping_rounds: int = 30
) -> Dict[str, Any]:
    """
    Full end-to-end XGBoost training and evaluation run.

    Early Stopping Strategy:
    --------------------------
    When use_early_stopping=True, the validation set is used for early stopping:
    - The XGBClassifier's internal eval_set is set to the preprocessed X_val array.
    - Training stops when the validation metric fails to improve for
      early_stopping_rounds consecutive rounds.
    - IMPORTANT: This does NOT violate train/val/test isolation. The validation
      set is ONLY used to determine when to stop — it is never used to update
      model weights. Test metrics are always computed on held-out test data.
    - The final model uses the round that achieved the best validation score,
      not necessarily all n_estimators rounds.

    Args:
        train_df: Raw training DataFrame.
        val_df: Raw validation DataFrame.
        test_df: Raw test DataFrame.
        hyperparams: Optional XGBoost hyperparameter overrides.
        model_dir: Directory to save model and config.
        include_city: Whether to include City as a feature.
        use_early_stopping: Whether to use early stopping on the validation set.
        early_stopping_rounds: Number of rounds without improvement before stopping.

    Returns:
        Dict[str, Any]: Results including pipeline, metrics, and timing.
    """
    from src.pipeline import prepare_all_splits

    version = check_xgboost()
    print(f"  XGBoost version: {version}")

    # Default hyperparameters
    hp = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "mlogloss",
        "n_classes": 6,
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

    X_train_df = data["X_train_df"]
    X_val_df   = data["X_val_df"]
    X_test_df  = data["X_test_df"]
    y_train    = data["y_train"]
    y_val      = data["y_val"]
    y_test     = data["y_test"]

    # Validate label range
    all_labels = set(y_train.unique()) | set(y_val.unique()) | set(y_test.unique())
    n_actual_classes = max(all_labels) + 1
    if n_actual_classes != hp["n_classes"]:
        print(f"  WARNING: Found {n_actual_classes} unique classes, "
              f"updating n_classes from {hp['n_classes']} to {n_actual_classes}.")
        hp["n_classes"] = int(n_actual_classes)

    preproc_pipeline = data["pipeline"]

    print("  Building XGBoost pipeline...")
    full_pipeline = build_xgboost_pipeline(preproc_pipeline, **hp)

    # Pre-transform data for early stopping eval_set
    # (early stopping requires numpy arrays, not DataFrames)
    X_train_t = preproc_pipeline.transform(X_train_df)
    X_val_t   = preproc_pipeline.transform(X_val_df)

    actual_early_stopping_rounds = None
    best_iteration = None

    if use_early_stopping:
        print(f"  Training XGBoost with early stopping "
              f"(patience={early_stopping_rounds} rounds)...")
        t_start = time.perf_counter()

        # Train with early stopping using pre-transformed arrays
        clf = full_pipeline.named_steps["classifier"]
        clf.set_params(early_stopping_rounds=early_stopping_rounds)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(
                X_train_t, y_train,
                eval_set=[(X_val_t, y_val)],
                verbose=False
            )

        train_time = round(time.perf_counter() - t_start, 4)
        best_iteration = clf.best_iteration
        actual_early_stopping_rounds = early_stopping_rounds
        print(f"  Training complete in {train_time:.2f}s. "
              f"Best iteration: {best_iteration}")

        # For consistent pipeline prediction, we use the fitted clf
        # but need a pipeline that preprocesses before predict
        # Re-wrap the trained clf into a predict-capable object
        full_pipeline.steps[-1] = ("classifier", clf)
    else:
        print(f"  Training XGBoost ({hp['n_estimators']} rounds, no early stopping)...")
        t_start = time.perf_counter()
        full_pipeline.fit(X_train_df, y_train)
        train_time = round(time.perf_counter() - t_start, 4)
        print(f"  Training complete in {train_time:.2f}s.")

    # Evaluate — using pre-transformed arrays for early-stopping runs,
    # or the full pipeline for standard runs
    if use_early_stopping:
        # Custom evaluation using pre-transformed arrays
        X_test_t = preproc_pipeline.transform(X_test_df)

        def _eval_on_array(clf_model, X_arr, y_arr, name):
            t0 = time.perf_counter()
            y_pred = clf_model.predict(X_arr)
            t_inf  = time.perf_counter() - t0
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
            present = sorted(y_arr.unique().tolist())
            labels  = [LABEL_TO_CATEGORY.get(i, str(i)) for i in present]
            return {
                "split": name,
                "n_samples": len(y_arr),
                "accuracy": float(accuracy_score(y_arr, y_pred)),
                "macro_precision": float(precision_score(y_arr, y_pred, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(y_arr, y_pred, average="macro", zero_division=0)),
                "macro_f1": float(f1_score(y_arr, y_pred, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_arr, y_pred, average="weighted", zero_division=0)),
                "per_class_report": classification_report(y_arr, y_pred, labels=present, target_names=labels, zero_division=0, output_dict=True),
                "confusion_matrix": confusion_matrix(y_arr, y_pred, labels=present).tolist(),
                "class_labels": labels,
                "inference_time_sec": round(t_inf, 4),
            }

        clf = full_pipeline.named_steps["classifier"]
        train_metrics = _eval_on_array(clf, X_train_t, y_train, "Train")
        val_metrics   = _eval_on_array(clf, X_val_t,   y_val,   "Validation")
        test_metrics  = _eval_on_array(clf, X_test_t,  y_test,  "Test")
    else:
        train_metrics = evaluate_model(full_pipeline, X_train_df, y_train, "Train")
        val_metrics   = evaluate_model(full_pipeline, X_val_df,   y_val,   "Validation")
        test_metrics  = evaluate_model(full_pipeline, X_test_df,  y_test,  "Test")

    print_evaluation_report(train_metrics)
    print_evaluation_report(val_metrics)
    print_evaluation_report(test_metrics)

    # Feature importances
    clf_model = full_pipeline.named_steps["classifier"]
    if hasattr(clf_model, "feature_importances_"):
        feat_names = data["feature_names_out"]
        importances = clf_model.feature_importances_
        n = min(len(feat_names), len(importances))
        df_imp = pd.DataFrame({
            "Feature": feat_names[:n],
            "Importance": importances[:n]
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        print("\n  Top 10 XGBoost Feature Importances:")
        print(df_imp.head(10).to_string(index=False))
    else:
        df_imp = pd.DataFrame()

    # Save model and config
    save_path = save_model_pipeline(full_pipeline, "xgboost", model_dir)
    config = {
        "model": "XGBClassifier",
        "xgboost_version": version,
        "hyperparameters": {k: str(v) for k, v in hp.items()},
        "feature_lists": data["feature_lists"],
        "feature_names_out": data["feature_names_out"],
        "train_time_sec": train_time,
        "early_stopping_rounds": actual_early_stopping_rounds,
        "best_iteration": best_iteration,
        "n_train": len(y_train),
        "n_val":   len(y_val),
        "n_test":  len(y_test),
        "train_accuracy": train_metrics["accuracy"],
        "val_accuracy":   val_metrics["accuracy"],
        "test_accuracy":  test_metrics["accuracy"],
        "val_macro_f1":   val_metrics["macro_f1"],
        "test_macro_f1":  test_metrics["macro_f1"],
    }
    config_path = save_model_config(config, "xgboost", model_dir)

    print(f"\n  Model saved to: {save_path}")
    print(f"  Config saved to: {config_path}")

    return {
        "pipeline": full_pipeline,
        "train_metrics": train_metrics,
        "val_metrics":   val_metrics,
        "test_metrics":  test_metrics,
        "train_time_sec": train_time,
        "best_iteration": best_iteration,
        "feature_importances": df_imp,
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

    results = run_xgboost(train_df, val_df, test_df, use_early_stopping=True)

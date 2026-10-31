"""
AirSense Support Vector Machine (SVM) Classifier Module
=========================================================
Implements SVM classification for AirSense using sklearn SVC integrated
with the AirSense preprocessing pipeline.

Support Vectors:
----------------
In a binary classification problem with two classes (+1 and −1), the SVM
finds the hyperplane that maximally separates the two classes. The training
examples that lie closest to the decision boundary are called support vectors.
Only these examples define the hyperplane — all other training examples
are irrelevant once the SVM is trained.

Hyperplanes and Margins:
------------------------
A hyperplane in P-dimensional space is a flat (P-1)-dimensional boundary:
    w · x + b = 0  (decision boundary)
    w · x + b = +1 (positive margin boundary)
    w · x + b = −1 (negative margin boundary)

The margin is the total width between the two margin boundaries: 2/||w||.

Maximum-Margin Classification:
--------------------------------
The SVM maximises the margin 2/||w|| subject to correct classification
of all training points. This is equivalent to minimising ||w||²/2.
Maximizing margin is a convex quadratic program with a unique global solution.

Why maximum margin? A wider margin provides greater geometric tolerance
for unseen test examples — intuitively, the classifier is "further from
the fence" and less likely to misclassify slightly shifted test points.

The Kernel Trick:
-----------------
Real data is rarely linearly separable. The kernel trick maps the input
features into a higher-dimensional (possibly infinite-dimensional) feature
space where a linear separator exists, without explicitly computing
the mapping φ(x):

    K(x_i, x_j) = φ(x_i) · φ(x_j)   [inner product in feature space]

Common kernels for AirSense:
1. Linear Kernel:  K(x_i, x_j) = x_i · x_j
   - No implicit mapping; equivalent to a standard linear classifier.
   - Fastest training. Useful when P is large or data is already linearly
     separable after scaling.

2. RBF (Radial Basis Function) Kernel:
   K(x_i, x_j) = exp(−γ ||x_i − x_j||²)
   - Creates infinite-dimensional mapping via Gaussian radial basis functions.
   - The parameter γ controls the radius of influence of each training example.
   - Small γ: smooth, wide influence (underfitting).
   - Large γ: narrow, local influence (overfitting).
   - Best for non-linear class boundaries (the typical case for air quality).

Why Scaling is Critical for SVM:
----------------------------------
SVM computes distances and inner products in feature space. A feature
with large numerical range (e.g., PM10 ∈ [0, 800]) will completely
dominate the distance computation over a feature with small range
(e.g., Benzene ∈ [0, 10]).

After StandardScaler, all features have mean=0 and std=1:
    PM10_scaled, Benzene_scaled ∈ similar ranges → fair comparison.

Without scaling, SVM performance is severely degraded. The AirSense
preprocessing pipeline always applies StandardScaler before SVM.

Soft-Margin SVM and the C Parameter:
--------------------------------------
Real data is never perfectly separable. The soft-margin SVM allows
a controlled number of margin violations using slack variables ξ_i:
    Minimize  ||w||²/2 + C Σ ξ_i
    Subject to  y_i(w·x_i + b) ≥ 1 − ξ_i   for all i

The regularization parameter C:
- Large C: Penalize misclassifications heavily → narrow margin, low bias,
  high variance (risk of overfitting).
- Small C: Tolerate misclassifications → wide margin, higher bias,
  lower variance.

Multiclass SVM:
---------------
sklearn's SVC handles multiclass via the One-vs-One (OvO) strategy:
For K=6 classes, it trains K(K-1)/2 = 15 binary SVMs, each separating
one class pair. The final prediction is the class that wins the most
pairwise duels.

Computational Limitations:
---------------------------
SVM training involves solving a quadratic program. Time complexity is
approximately O(N²) to O(N³) where N is the number of training samples.
With N=17,394 training rows, full SVM (RBF) is feasible but slower
than tree models.

For the full city_day.csv training set of ~17K rows, RBF SVM typically
takes 30s–10 minutes depending on C, gamma, and the number of support vectors.
If training time is impractical, a justified sub-sample can be used with
honest disclosure.
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.svm import SVC
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


def build_svm_pipeline(
    preprocessing_pipeline: Pipeline,
    kernel: str = "rbf",
    C: float = 10.0,
    gamma: str = "scale",
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
    probability: bool = False,
    decision_function_shape: str = "ovr"
) -> Pipeline:
    """
    Build a Pipeline combining the AirSense preprocessor with SVC.

    Why these defaults?
    -------------------
    - kernel='rbf': RBF is the standard first-choice kernel for tabular
      data. It can model non-linear pollutant-AQI class boundaries that
      a linear kernel cannot.
    - C=10.0: Moderate penalty — allows some margin violations. Prevents
      overfitting on noisy sensor data while still fitting the training set.
    - gamma='scale': Uses 1/(n_features × X.var()) — adapts to the
      feature count and variance automatically after StandardScaler.
    - class_weight='balanced': Essential for CPCB imbalanced classes
      (Severe=5.4%, Moderate=35.5%).
    - probability=False: Computing class probabilities (via Platt scaling)
      requires an extra 5-fold internal cross-validation pass and is
      computationally expensive. Disabled by default; enable if needed.
    - decision_function_shape='ovr': One-vs-Rest decision function shape
      for more stable multiclass probability estimates.

    Args:
        preprocessing_pipeline: AirSense preprocessing Pipeline.
        kernel: SVM kernel type ('rbf', 'linear', 'poly').
        C: Regularization parameter.
        gamma: Kernel coefficient for 'rbf'/'poly'/'sigmoid'.
        class_weight: Class weighting for imbalanced classes.
        random_state: Seed for reproducibility.
        probability: Whether to enable probability estimates.
        decision_function_shape: 'ovr' or 'ovo'.

    Returns:
        sklearn.pipeline.Pipeline: Full Pipeline (preprocessing + SVC).
    """
    classifier = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        class_weight=class_weight,
        random_state=random_state,
        probability=probability,
        decision_function_shape=decision_function_shape,
        cache_size=500  # 500MB kernel cache for speed
    )

    full_pipeline = Pipeline(
        steps=preprocessing_pipeline.steps + [("classifier", classifier)]
    )
    return full_pipeline


def run_svm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    hyperparams: Optional[Dict[str, Any]] = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
    include_city: bool = False,
    max_train_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Full end-to-end SVM training and evaluation run.

    IMPORTANT — Computational Cost Warning:
    ----------------------------------------
    RBF SVM training time is O(N²) to O(N³). For N≈17K training rows,
    expect 2–15 minutes. If max_train_samples is set, a random stratified
    subsample is used and the subsampling is explicitly disclosed in the
    config and evaluation report.

    Args:
        train_df: Raw training DataFrame.
        val_df: Raw validation DataFrame.
        test_df: Raw test DataFrame.
        hyperparams: Optional SVM hyperparameter overrides.
        model_dir: Directory to save model and config.
        include_city: Whether to include City as a feature.
        max_train_samples: If set, subsample training to this many rows.
                           Classes are subsampled proportionally (stratified).
                           Evaluation is still performed on the full val/test sets.

    Returns:
        Dict[str, Any]: Results including pipeline, metrics, and timing.
    """
    from src.pipeline import prepare_all_splits

    # Default hyperparameters
    hp = {
        "kernel": "rbf",
        "C": 10.0,
        "gamma": "scale",
        "class_weight": "balanced",
        "random_state": 42,
        "probability": False,
        "decision_function_shape": "ovr",
    }
    if hyperparams:
        hp.update(hyperparams)

    print("  Preparing data splits...")
    data = prepare_all_splits(
        train_df, val_df, test_df,
        include_city=include_city,
        scale_numerical=True,  # CRITICAL for SVM
        winsorize=True
    )

    X_train_df = data["X_train_df"]
    X_val_df   = data["X_val_df"]
    X_test_df  = data["X_test_df"]
    y_train    = data["y_train"]
    y_val      = data["y_val"]
    y_test     = data["y_test"]

    # Optional: stratified subsampling for computational feasibility
    subsampled = False
    original_n_train = len(y_train)
    if max_train_samples and len(y_train) > max_train_samples:
        print(f"  NOTICE: Subsampling training set from {len(y_train):,} to "
              f"{max_train_samples:,} rows (stratified by class).")
        print(f"  Evaluation will use the FULL validation and test sets.")

        # Stratified subsample
        rng = np.random.RandomState(hp["random_state"])
        subsample_idx = []
        for cls in y_train.unique():
            cls_idx = y_train[y_train == cls].index.tolist()
            n_cls = max(1, int(round(max_train_samples * len(cls_idx) / len(y_train))))
            chosen = rng.choice(cls_idx, size=min(n_cls, len(cls_idx)), replace=False)
            subsample_idx.extend(chosen.tolist())

        rng.shuffle(subsample_idx)
        X_train_df = X_train_df.loc[subsample_idx].reset_index(drop=True)
        y_train    = y_train.loc[subsample_idx].reset_index(drop=True)
        subsampled = True
        print(f"  Subsample sizes per class:")
        for cls, cnt in y_train.value_counts().sort_index().items():
            label = LABEL_TO_CATEGORY.get(cls, str(cls))
            print(f"    {label}: {cnt:,}")

    preproc_pipeline = data["pipeline"]

    print("  Building SVM pipeline...")
    full_pipeline = build_svm_pipeline(preproc_pipeline, **hp)

    print(f"  Training SVM (kernel={hp['kernel']}, C={hp['C']}, gamma={hp['gamma']}) "
          f"on {len(y_train):,} samples...")
    print("  (RBF SVM training can take several minutes on large datasets)")
    t_start = time.perf_counter()
    full_pipeline.fit(X_train_df, y_train)
    train_time = round(time.perf_counter() - t_start, 4)
    print(f"  Training complete in {train_time:.2f}s.")

    # Evaluate
    train_metrics = evaluate_model(full_pipeline, X_train_df, y_train, "Train (subset)" if subsampled else "Train")
    val_metrics   = evaluate_model(full_pipeline, X_val_df,   y_val,   "Validation")
    test_metrics  = evaluate_model(full_pipeline, X_test_df,  y_test,  "Test")

    if subsampled:
        print(f"\n  NOTE: Train metrics reflect the {len(y_train):,}-sample subset,")
        print(f"  not the full {original_n_train:,}-row training set.")

    print_evaluation_report(train_metrics)
    print_evaluation_report(val_metrics)
    print_evaluation_report(test_metrics)

    # Save model and config
    save_path = save_model_pipeline(full_pipeline, "svm", model_dir)
    config = {
        "model": "SVC",
        "hyperparameters": hp,
        "feature_lists": data["feature_lists"],
        "feature_names_out": data["feature_names_out"],
        "train_time_sec": train_time,
        "subsampled": subsampled,
        "original_n_train": original_n_train,
        "used_n_train": len(y_train),
        "n_val":  len(y_val),
        "n_test": len(y_test),
        "train_accuracy": train_metrics["accuracy"],
        "val_accuracy":   val_metrics["accuracy"],
        "test_accuracy":  test_metrics["accuracy"],
        "val_macro_f1":   val_metrics["macro_f1"],
        "test_macro_f1":  test_metrics["macro_f1"],
    }
    config_path = save_model_config(config, "svm", model_dir)

    print(f"\n  Model saved to: {save_path}")
    print(f"  Config saved to: {config_path}")

    return {
        "pipeline": full_pipeline,
        "train_metrics": train_metrics,
        "val_metrics":   val_metrics,
        "test_metrics":  test_metrics,
        "train_time_sec": train_time,
        "subsampled": subsampled,
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

    # Use max_train_samples for initial feasibility check; remove for full training
    results = run_svm(train_df, val_df, test_df, max_train_samples=5000)

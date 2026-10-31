"""
AirSense Decision Tree Classifier Module
==========================================
Implements a baseline Decision Tree classifier for AirSense air quality
classification. Uses sklearn DecisionTreeClassifier integrated with the
AirSense preprocessing pipeline.

What is a Decision Tree?
------------------------
A Decision Tree is a supervised learning algorithm that makes predictions
by learning simple decision rules from training features. It partitions
the feature space into rectangular regions, one question at a time:

  "Is PM2.5 > 60?"  →  YES → "Is Month == Monsoon?"  →  YES → Poor
                     →  NO  → "Is CO > 1.5?"          →  NO  → Good

Anatomy of a Decision Tree:
-----------------------------
- Root Node: The very first split — the single feature/threshold that
  best separates the training classes.
- Internal Nodes: Intermediate decision points that further partition
  each sub-region.
- Leaf Nodes: Terminal nodes that output the predicted class label
  (the majority class in that leaf's training samples).

Splitting Criteria:
-------------------
At each node, the algorithm evaluates every possible (feature, threshold)
pair and picks the split that minimizes impurity in the resulting children.

1. Gini Impurity (default):
   Gini(t) = 1 − Σ p_k²
   where p_k = fraction of class k samples at node t.
   A pure node (only one class) has Gini = 0.
   A maximally mixed node (equal classes) has Gini approaching 1.
   Example: Node with [Good=10, Severe=10] → Gini = 1 − (0.5²+0.5²) = 0.50.
   After split: [Good=10, Severe=0] + [Good=0, Severe=10] → weighted Gini = 0.

2. Entropy (Information Gain):
   H(t) = −Σ p_k log₂(p_k)
   Information Gain = H(parent) − Σ (n_child/n_parent) × H(child)
   Selects splits that maximally reduce uncertainty.

For air quality: if PM2.5 > 150 separates 'Severe' observations nearly
perfectly, that split gets a very high information gain.

Overfitting and Pruning:
------------------------
Without constraints, a Decision Tree will grow until every leaf contains
exactly one training sample, memorizing noise. Remedies:
- max_depth: Limit tree depth (e.g., max_depth=5 → at most 32 leaves).
- min_samples_split: Require at least N samples to attempt a split.
- min_samples_leaf: Require at least N samples to form a leaf.
- ccp_alpha: Cost-complexity pruning — remove sub-trees that add little
  predictive value relative to their added complexity.

Advantages for AirSense:
-------------------------
- Interpretable: Can be visualized and audited rule-by-rule.
- No scaling required: Trees are threshold-based, not distance-based.
- Handles mixed feature types naturally.
- Naturally multi-class.

Limitations:
------------
- High variance: Small data changes can produce very different trees.
- Rectangular boundaries: Cannot model diagonal or curved class boundaries.
- Tends to over-fit on training data without pruning.
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from src.feature_engineering import LEAKAGE_COLS
from src.preprocessing import LABEL_TO_CATEGORY, CATEGORY_TO_LABEL
from src.data_loader import CPCB_AQI_CATEGORIES

# Default output directory for saved models
DEFAULT_MODEL_DIR = Path("models")


def build_decision_tree_pipeline(
    preprocessing_pipeline: Pipeline,
    max_depth: Optional[int] = 10,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
    criterion: str = "gini",
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
    ccp_alpha: float = 0.0
) -> Pipeline:
    """
    Build a scikit-learn Pipeline that combines the AirSense preprocessing
    pipeline with a DecisionTreeClassifier.

    Why these defaults?
    -------------------
    - max_depth=10: Limits overfitting while allowing meaningful splits.
    - min_samples_split=20: Prevents splitting on tiny groups (noise).
    - min_samples_leaf=10: Each leaf must represent at least 10 samples.
    - class_weight='balanced': Corrects for class imbalance (Severe=5.4%,
      Moderate=35.5%). Without this, the tree would bias toward Moderate.
    - criterion='gini': Standard Gini impurity — computationally efficient
      and empirically effective.
    - random_state=42: Ensures reproducible tree structure.

    Args:
        preprocessing_pipeline: Fitted or unfitted AirSense preprocessing
                                 Pipeline from src.pipeline.
        max_depth: Maximum tree depth. None = unlimited (risk of overfitting).
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required at each leaf node.
        criterion: Split quality metric ('gini' or 'entropy').
        class_weight: 'balanced', dict, or None. 'balanced' recommended
                      for AirSense due to class imbalance.
        random_state: Random seed for reproducibility.
        ccp_alpha: Complexity parameter for Minimal Cost-Complexity Pruning.

    Returns:
        sklearn.pipeline.Pipeline: Full Pipeline (preprocessing + classifier).
    """
    classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        class_weight=class_weight,
        random_state=random_state,
        ccp_alpha=ccp_alpha
    )

    # Append classifier to the preprocessing pipeline
    full_pipeline = Pipeline(
        steps=preprocessing_pipeline.steps + [("classifier", classifier)]
    )
    return full_pipeline


def train_decision_tree(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[Pipeline, float]:
    """
    Fit the full Decision Tree pipeline on training data.

    Args:
        pipeline: Unfitted full Pipeline from build_decision_tree_pipeline().
        X_train: Training feature DataFrame.
        y_train: Integer-encoded training target Series.

    Returns:
        Tuple[Pipeline, float]: (fitted_pipeline, training_time_seconds)
    """
    t_start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    t_end = time.perf_counter()
    return pipeline, round(t_end - t_start, 4)


def evaluate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y_true: pd.Series,
    split_name: str = "Validation"
) -> Dict[str, Any]:
    """
    Evaluate a fitted Pipeline on a labeled dataset split.

    Metrics reported:
    -----------------
    - Accuracy: Fraction of correctly classified observations.
    - Macro Precision: Average precision across all 6 classes (equal weight).
    - Macro Recall: Average recall across all 6 classes.
    - Macro F1: Harmonic mean of macro precision and recall.
    - Weighted F1: F1 weighted by class support (sample count per class).
    - Per-class Precision, Recall, F1: Detailed breakdowns per CPCB category.
    - Confusion Matrix: N×N matrix showing predicted vs. actual counts.

    Args:
        pipeline: Fitted Pipeline.
        X: Feature DataFrame.
        y_true: True integer labels.
        split_name: Label for reporting (e.g., 'Train', 'Validation', 'Test').

    Returns:
        Dict[str, Any]: Comprehensive evaluation metrics dictionary.
    """
    t_start = time.perf_counter()
    y_pred = pipeline.predict(X)
    t_inf = time.perf_counter() - t_start

    # Determine which classes are present in y_true
    present_classes = sorted(y_true.unique().tolist())
    class_labels = [LABEL_TO_CATEGORY.get(i, str(i)) for i in present_classes]

    accuracy = float(accuracy_score(y_true, y_pred))

    macro_prec = float(precision_score(
        y_true, y_pred, average="macro", zero_division=0
    ))
    macro_rec = float(recall_score(
        y_true, y_pred, average="macro", zero_division=0
    ))
    macro_f1 = float(f1_score(
        y_true, y_pred, average="macro", zero_division=0
    ))
    weighted_f1 = float(f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    ))

    # Per-class report
    report_dict = classification_report(
        y_true, y_pred,
        labels=present_classes,
        target_names=class_labels,
        zero_division=0,
        output_dict=True
    )

    cm = confusion_matrix(y_true, y_pred, labels=present_classes)

    return {
        "split": split_name,
        "n_samples": len(y_true),
        "accuracy": accuracy,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "class_labels": class_labels,
        "inference_time_sec": round(t_inf, 4),
    }


def save_model_pipeline(
    pipeline: Pipeline,
    model_name: str = "decision_tree",
    model_dir: Path = DEFAULT_MODEL_DIR
) -> Path:
    """
    Save the fitted Pipeline to disk using joblib.

    Args:
        pipeline: Fitted sklearn Pipeline.
        model_name: Base name for the saved file.
        model_dir: Directory to save into.

    Returns:
        Path: Saved file path.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / f"{model_name}.joblib"
    joblib.dump(pipeline, save_path)
    return save_path


def save_model_config(
    config: Dict[str, Any],
    model_name: str = "decision_tree",
    model_dir: Path = DEFAULT_MODEL_DIR
) -> Path:
    """Save model configuration as a JSON file."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir / f"{model_name}_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    return config_path


def print_evaluation_report(metrics: Dict[str, Any]) -> None:
    """
    Print a human-readable classification report to stdout.
    """
    print(f"\n{'='*60}")
    print(f"  Evaluation Report — {metrics['split']}")
    print(f"{'='*60}")
    print(f"  Samples       : {metrics['n_samples']:,}")
    print(f"  Accuracy      : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall   : {metrics['macro_recall']:.4f}")
    print(f"  Macro F1       : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1    : {metrics['weighted_f1']:.4f}")
    print(f"  Inference Time : {metrics['inference_time_sec']:.4f}s")
    print()

    print("  Per-Class Metrics:")
    print(f"  {'Category':<16} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*56}")
    for cls in metrics["class_labels"]:
        r = metrics["per_class_report"].get(cls, {})
        prec = r.get("precision", 0.0)
        rec  = r.get("recall", 0.0)
        f1   = r.get("f1-score", 0.0)
        sup  = r.get("support", 0)
        print(f"  {cls:<16} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {int(sup):>10,}")

    print()
    print("  Confusion Matrix:")
    cm = np.array(metrics["confusion_matrix"])
    labels = metrics["class_labels"]
    header = "  " + "".join(f"{l[:8]:>10}" for l in labels)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  " + f"{labels[i][:8]:<10}" + "".join(f"{v:>10}" for v in row)
        print(row_str)
    print()


def run_decision_tree(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    hyperparams: Optional[Dict[str, Any]] = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
    include_city: bool = False
) -> Dict[str, Any]:
    """
    Full end-to-end Decision Tree training and evaluation run.

    Args:
        train_df: Raw training DataFrame.
        val_df: Raw validation DataFrame.
        test_df: Raw test DataFrame.
        hyperparams: Optional dict of Decision Tree hyperparameter overrides.
        model_dir: Directory to save trained model and config.
        include_city: Whether to include City as a predictor feature.

    Returns:
        Dict[str, Any]: Results including fitted pipeline and all metrics.
    """
    from src.pipeline import (
        build_preprocessing_pipeline,
        prepare_all_splits,
    )

    # Default hyperparameters
    hp = {
        "max_depth": 10,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "criterion": "gini",
        "class_weight": "balanced",
        "random_state": 42,
        "ccp_alpha": 0.0,
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

    # Build full pipeline (preprocessing already fitted inside prepare_all_splits,
    # but here we build a fresh classifier pipeline for separate fit call)
    print("  Building Decision Tree pipeline...")
    preproc_pipeline = data["pipeline"]

    # Build fresh end-to-end pipeline (preprocess + classifier)
    full_pipeline = build_decision_tree_pipeline(
        preproc_pipeline, **hp
    )

    # Refit on raw X_train (pipeline will refit preprocessing internally)
    print("  Training Decision Tree...")
    X_train_df = data["X_train_df"]
    X_val_df   = data["X_val_df"]
    X_test_df  = data["X_test_df"]
    y_train = data["y_train"]
    y_val   = data["y_val"]
    y_test  = data["y_test"]

    full_pipeline, train_time = train_decision_tree(full_pipeline, X_train_df, y_train)
    print(f"  Training complete in {train_time:.2f}s.")

    # Evaluate
    train_metrics = evaluate_model(full_pipeline, X_train_df, y_train, "Train")
    val_metrics   = evaluate_model(full_pipeline, X_val_df,   y_val,   "Validation")
    test_metrics  = evaluate_model(full_pipeline, X_test_df,  y_test,  "Test")

    print_evaluation_report(train_metrics)
    print_evaluation_report(val_metrics)
    print_evaluation_report(test_metrics)

    # Save model and config
    save_path   = save_model_pipeline(full_pipeline, "decision_tree", model_dir)
    config = {
        "model": "DecisionTreeClassifier",
        "hyperparameters": hp,
        "feature_lists": data["feature_lists"],
        "feature_names_out": data["feature_names_out"],
        "train_time_sec": train_time,
        "n_train": len(y_train),
        "n_val":   len(y_val),
        "n_test":  len(y_test),
        "train_accuracy": train_metrics["accuracy"],
        "val_accuracy":   val_metrics["accuracy"],
        "test_accuracy":  test_metrics["accuracy"],
        "val_macro_f1":   val_metrics["macro_f1"],
        "test_macro_f1":  test_metrics["macro_f1"],
    }
    config_path = save_model_config(config, "decision_tree", model_dir)

    print(f"  Model saved to: {save_path}")
    print(f"  Config saved to: {config_path}")

    return {
        "pipeline": full_pipeline,
        "train_metrics": train_metrics,
        "val_metrics":   val_metrics,
        "test_metrics":  test_metrics,
        "train_time_sec": train_time,
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

    results = run_decision_tree(train_df, val_df, test_df)

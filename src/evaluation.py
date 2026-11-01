"""
AirSense Unified Evaluation Module
====================================
Centralised evaluation library used by all AirSense model scripts.

Responsibilities
----------------
- compute_metrics()           : Compute all 9 metric groups from y_true / y_pred.
- build_comparison_table()    : Assemble a summary DataFrame across all models.
- select_best_model()         : Choose the winning model with a documented criterion.
- plot_performance_comparison(): 5-metric bar chart across all models.
- plot_macro_f1_comparison()  : Val vs Test macro F1 bar chart.
- plot_confusion_matrix()     : Styled heatmap per model.
- plot_per_class_performance(): Grouped per-class F1 chart across all models.

Design rules
------------
- Every metric is derived from actual predict() output — no hard-coded values.
- The output directory is created automatically if it does not exist.
- All chart functions return the figure path so callers can log or embed it.
- If a model result dict contains 'error', it is excluded from charts with a
  clear console warning rather than crashing the whole evaluation run.
"""

import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # non-interactive backend for script execution
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_loader import CPCB_AQI_CATEGORIES
from src.preprocessing import LABEL_TO_CATEGORY

# ─────────────────────────────────────────────────────────────────
# Shared colour palette (consistent across all charts)
# ─────────────────────────────────────────────────────────────────

MODEL_COLOURS: Dict[str, str] = {
    "Decision Tree":  "#4C72B0",
    "Random Forest":  "#DD8452",
    "SVM":            "#55A868",
    "XGBoost":        "#C44E52",
}

METRIC_COLOURS: Dict[str, str] = {
    "Accuracy":          "#4C72B0",
    "Macro Precision":   "#DD8452",
    "Macro Recall":      "#55A868",
    "Macro F1":          "#C44E52",
    "Weighted F1":       "#8172B2",
}

# Seaborn / Matplotlib global style
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ─────────────────────────────────────────────────────────────────
# 1. Metric Computation
# ─────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    split_name: str = "Test",
    inference_time_sec: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics from actual label arrays.

    Metrics returned
    ----------------
    - accuracy           : Overall fraction correct.
    - macro_precision    : Unweighted average precision across 6 classes.
    - macro_recall       : Unweighted average recall across 6 classes.
    - macro_f1           : Macro-averaged F1.
    - weighted_f1        : Support-weighted F1.
    - per_class_report   : sklearn classification_report dict (per-class P/R/F1).
    - confusion_matrix   : N×N integer array (list-of-lists for JSON serialisation).
    - class_labels       : List of CPCB category strings corresponding to matrix rows/cols.
    - inference_time_sec : Wall-clock seconds for the full predict() call.

    Args
    ----
    y_true             : Ground-truth integer labels (0–5).
    y_pred             : Predicted integer labels from model.predict().
    split_name         : Descriptive name of the data split.
    inference_time_sec : Pre-measured prediction latency in seconds.

    Returns
    -------
    Dict[str, Any]: Complete metrics dictionary.
    """
    y_true_arr = np.asarray(y_true)
    if not len(y_true_arr) or len(y_true_arr) != len(y_pred):
        raise ValueError("Expected equally sized, nonempty target and prediction arrays")
    if not set(np.unique(y_true_arr)).union(np.unique(y_pred)).issubset(set(range(6))):
        raise ValueError("Predictions and targets must use CPCB integer labels 0 through 5")
    present_classes = list(range(6))  # Fixed CPCB label universe, including absent classes
    class_labels = [LABEL_TO_CATEGORY.get(i, str(i)) for i in present_classes]

    accuracy      = float(accuracy_score(y_true_arr, y_pred))
    macro_prec    = float(precision_score(y_true_arr, y_pred, average="macro",    labels=present_classes, zero_division=0))
    macro_rec     = float(recall_score   (y_true_arr, y_pred, average="macro",    labels=present_classes, zero_division=0))
    macro_f1      = float(f1_score       (y_true_arr, y_pred, average="macro",    labels=present_classes, zero_division=0))
    weighted_f1   = float(f1_score       (y_true_arr, y_pred, average="weighted", labels=present_classes, zero_division=0))

    report_dict = classification_report(
        y_true_arr, y_pred,
        labels=present_classes,
        target_names=class_labels,
        zero_division=0,
        output_dict=True,
    )

    cm = confusion_matrix(y_true_arr, y_pred, labels=present_classes)

    return {
        "split":             split_name,
        "n_samples":         int(len(y_true_arr)),
        "accuracy":          round(accuracy,    6),
        "macro_precision":   round(macro_prec,  6),
        "macro_recall":      round(macro_rec,   6),
        "macro_f1":          round(macro_f1,    6),
        "weighted_f1":       round(weighted_f1, 6),
        "per_class_report":  report_dict,
        "confusion_matrix":  cm.tolist(),
        "class_labels":      class_labels,
        "inference_time_sec": round(inference_time_sec, 6),
    }


# ─────────────────────────────────────────────────────────────────
# 2. Summary Table Builder
# ─────────────────────────────────────────────────────────────────

def build_comparison_table(
    all_results: Dict[str, Dict[str, Any]],
    split: str = "Test",
) -> pd.DataFrame:
    """
    Build a flat summary DataFrame from all model result dicts.

    Args
    ----
    all_results : Dict mapping display model name → result dict from evaluate_all_models.
                  Each result dict must contain 'test_metrics' and 'train_time_sec'.
    split       : Which split's metrics to include ('Test' or 'Validation').

    Returns
    -------
    pd.DataFrame with columns:
        model, accuracy, macro_precision, macro_recall, macro_f1,
        weighted_f1, train_time_sec, inference_time_sec
    """
    rows = []
    split_key = f"{split.lower()}_metrics"

    for model_name, result in all_results.items():
        if "error" in result:
            print(f"  [SKIP] {model_name} has an error entry — excluded from table: {result['error']}")
            continue

        m = result.get(split_key, {})
        rows.append({
            "model":              model_name,
            "accuracy":           m.get("accuracy",          float("nan")),
            "macro_precision":    m.get("macro_precision",   float("nan")),
            "macro_recall":       m.get("macro_recall",      float("nan")),
            "macro_f1":           m.get("macro_f1",          float("nan")),
            "weighted_f1":        m.get("weighted_f1",       float("nan")),
            "train_time_sec":     result.get("train_time_sec",     float("nan")),
            "inference_time_sec": m.get("inference_time_sec", float("nan")),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────
# 3. Best Model Selection
# ─────────────────────────────────────────────────────────────────

def select_best_model(
    comparison_df: pd.DataFrame,
    criterion: str = "macro_f1",
    split: str = "Validation",
) -> Tuple[str, str]:
    """
    Select the best model according to a documented criterion.

    Criterion rationale
    -------------------
    macro_f1 (default)  : Unweighted average F1 across all 6 CPCB classes.
        Prioritised over accuracy because the dataset has significant class
        imbalance (Moderate ~35%, Severe/Good ~5% each). A model that ignores
        rare classes can achieve high accuracy with low macro F1. Macro F1
        forces equal weight on every category, including health-critical extremes.

    Args
    ----
    comparison_df : DataFrame from build_comparison_table().
    criterion     : Column name to rank models by (descending).

    Returns
    -------
    Tuple[str, str]: (best_model_name, justification_string)
    """
    if split.lower() != "validation":
        raise ValueError("Model selection requires validation metrics; test scores cannot select a model")
    if comparison_df.empty:
        return ("N/A", "No results available.")

    if criterion not in comparison_df.columns:
        criterion = "macro_f1"

    best_row = comparison_df.sort_values(criterion, ascending=False).iloc[0]
    best_name = best_row["model"]
    best_val  = best_row[criterion]

    justification = (
        f"'{best_name}' selected as best model based on **{criterion}** = {best_val:.4f} "
        f"on the validation set.\n\n"
        f"Criterion justification: Macro F1 (unweighted average F1 across all 6 CPCB classes) "
        f"is the primary selection metric because the AirSense dataset exhibits significant "
        f"class imbalance (Moderate ≈35%, Severe/Good ≈5%). "
        f"Accuracy alone rewards models that ignore rare health-critical categories. "
        f"Macro F1 ensures equal weight on every class, including 'Good' (clean air events) "
        f"and 'Severe' (public health emergencies). "
        f"Validation macro F1 was used for hyperparameter selection; "
        f"the selected candidate is frozen before final test evaluation."
    )

    return (best_name, justification)


# ─────────────────────────────────────────────────────────────────
# 4. Plotting Functions
# ─────────────────────────────────────────────────────────────────

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_performance_comparison(
    comparison_df: pd.DataFrame,
    out_dir: Path,
    filename: str = "performance_comparison.png",
) -> Path:
    """
    Generate a grouped bar chart comparing 5 summary metrics across all models.

    Metrics plotted: Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1.
    All values come from the comparison_df produced by build_comparison_table().
    """
    _ensure_dir(out_dir)

    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    metric_labels = ["Accuracy", "Macro\nPrecision", "Macro\nRecall", "Macro\nF1", "Weighted\nF1"]
    models = comparison_df["model"].tolist()
    n_metrics = len(metrics)
    n_models  = len(models)

    x = np.arange(n_metrics)
    bar_width = 0.18
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_width

    colours = list(MODEL_COLOURS.values())[:n_models]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (model_name, colour) in enumerate(zip(models, colours)):
        row = comparison_df[comparison_df["model"] == model_name].iloc[0]
        values = [row[m] for m in metrics]
        bars = ax.bar(x + offsets[i], values, bar_width * 0.9, label=model_name,
                      color=colour, alpha=0.88, edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=6.5, rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("AirSense — Model Performance Comparison (Test Set)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(title="Model", loc="lower right", fontsize=9, title_fontsize=9)
    ax.grid(axis="y", alpha=0.4)

    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [CHART] Saved: {out_path}")
    return out_path


def plot_macro_f1_comparison(
    comparison_df: pd.DataFrame,
    val_results: Optional[Dict[str, Any]],
    out_dir: Path,
    filename: str = "macro_f1_comparison.png",
) -> Path:
    """
    Generate a grouped bar chart of Validation vs Test Macro F1 per model.

    Having both bars side-by-side makes over/under-fitting immediately visible.
    """
    _ensure_dir(out_dir)

    models     = comparison_df["model"].tolist()
    test_f1s   = comparison_df["macro_f1"].tolist()
    val_f1s    = []

    for model_name in models:
        if val_results and model_name in val_results:
            vr = val_results[model_name]
            vm = vr.get("val_metrics") or vr.get("validation_metrics", {})
            val_f1s.append(vm.get("macro_f1", float("nan")))
        else:
            val_f1s.append(float("nan"))

    x = np.arange(len(models))
    bar_width = 0.35
    colours = list(MODEL_COLOURS.values())[:len(models)]

    fig, ax = plt.subplots(figsize=(10, 5))

    bars_val  = ax.bar(x - bar_width / 2, val_f1s,  bar_width, label="Validation Macro F1",
                       color=[c + "99" for c in colours], edgecolor=colours, linewidth=1.2)
    bars_test = ax.bar(x + bar_width / 2, test_f1s, bar_width, label="Test Macro F1",
                       color=colours, edgecolor="white", linewidth=0.6)

    for bar, val in list(zip(bars_val, val_f1s)) + list(zip(bars_test, test_f1s)):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Macro F1-Score", fontsize=11)
    ax.set_title("AirSense — Macro F1 Score: Validation vs Test", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.4)

    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [CHART] Saved: {out_path}")
    return out_path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_labels: List[str],
    model_name: str,
    out_dir: Path,
    filename_suffix: str = "",
) -> Path:
    """
    Generate a styled confusion matrix heatmap for a single model.

    The diagonal is highlighted in a distinct colour to make correct
    classifications stand out. Row normalisation percentages are annotated
    inside cells alongside raw counts.
    """
    _ensure_dir(out_dir)

    cm_arr = np.array(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(
            cm_arr.sum(axis=1, keepdims=True) > 0,
            cm_arr / cm_arr.sum(axis=1, keepdims=True),
            0.0,
        )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=False,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Row-normalised fraction"},
    )

    # Annotate each cell with raw count + percentage
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            raw  = int(cm_arr[i, j])
            pct  = cm_norm[i, j] * 100
            text_color = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(
                j + 0.5, i + 0.5,
                f"{raw}\n({pct:.1f}%)",
                ha="center", va="center",
                fontsize=7.5, color=text_color,
                fontweight="bold" if i == j else "normal",
            )

    safe_name = model_name.lower().replace(" ", "_")
    ax.set_title(f"Confusion Matrix — {model_name}\n(row-normalised, test set)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)

    fig.tight_layout()
    filename = f"confusion_matrix_{safe_name}{filename_suffix}.png"
    out_path = out_dir / filename
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [CHART] Saved: {out_path}")
    return out_path


def plot_per_class_performance(
    all_results: Dict[str, Dict[str, Any]],
    out_dir: Path,
    filename: str = "per_class_performance.png",
    metric: str = "f1-score",
) -> Path:
    """
    Generate a grouped bar chart of per-class F1 scores across all models.

    Each group on the x-axis is one CPCB AQI class. Each bar within the
    group represents one model. This makes it easy to identify which
    classes are universally difficult vs. model-specific.
    """
    _ensure_dir(out_dir)

    classes = CPCB_AQI_CATEGORIES  # canonical order: Good → Severe
    model_names = [
        m for m in all_results
        if "error" not in all_results[m]
    ]

    data: Dict[str, List[float]] = {model: [] for model in model_names}

    for model_name in model_names:
        result  = all_results[model_name]
        metrics = result.get("test_metrics", {})
        report  = metrics.get("per_class_report", {})
        for cls in classes:
            cls_data = report.get(cls, {})
            data[model_name].append(float(cls_data.get(metric, float("nan"))))

    x = np.arange(len(classes))
    n_models  = len(model_names)
    bar_width = 0.18
    offsets   = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_width
    colours   = list(MODEL_COLOURS.values())[:n_models]

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (model_name, colour) in enumerate(zip(model_names, colours)):
        vals = data[model_name]
        ax.bar(
            x + offsets[i], vals, bar_width * 0.9,
            label=model_name, color=colour, alpha=0.88,
            edgecolor="white", linewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylabel(f"Per-class {metric.replace('-', ' ').title()}", fontsize=11)
    ax.set_title(
        f"AirSense — Per-Class {metric.replace('-', ' ').title()} by Model (Test Set)",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax.set_ylim(0, 1.10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(title="Model", loc="lower left", fontsize=9, title_fontsize=9)
    ax.grid(axis="y", alpha=0.4)

    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [CHART] Saved: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────
# 5. Console Report Printer
# ─────────────────────────────────────────────────────────────────

def print_model_summary(model_name: str, result: Dict[str, Any]) -> None:
    """Print a compact human-readable summary for one model."""
    if "error" in result:
        print(f"\n  [{model_name}] FAILED: {result['error']}")
        return

    m = result.get("test_metrics", {})
    print(f"\n  {'='*58}")
    print(f"  Model         : {model_name}")
    print(f"  Train time    : {result.get('train_time_sec', float('nan')):.2f}s")
    print(f"  Test samples  : {m.get('n_samples', '?'):,}")
    print(f"  Accuracy      : {m.get('accuracy', float('nan')):.4f}")
    print(f"  Macro F1      : {m.get('macro_f1', float('nan')):.4f}")
    print(f"  Weighted F1   : {m.get('weighted_f1', float('nan')):.4f}")
    print(f"  Inference time: {m.get('inference_time_sec', float('nan')):.4f}s")
    print()

    report = m.get("per_class_report", {})
    labels = m.get("class_labels", CPCB_AQI_CATEGORIES)
    print(f"  {'Category':<16} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*56}")
    for cls in labels:
        r   = report.get(cls, {})
        prec = r.get("precision", float("nan"))
        rec  = r.get("recall",    float("nan"))
        f1   = r.get("f1-score",  float("nan"))
        sup  = r.get("support",   0)
        print(f"  {cls:<16} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {int(sup):>10,}")
    print()

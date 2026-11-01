"""
AirSense Feature Importance & Explainability Module
=====================================================
Provides four complementary methods for understanding which features
drive AirSense model predictions:

1. MDI (Mean Decrease in Impurity) — `feature_importances_` from tree ensembles.
   Fast, computed during training. Biased toward high-cardinality features.

2. Permutation Importance — shuffles each feature on a held-out split and
   measures the drop in macro F1. Model-agnostic, less biased than MDI.
   Applied on the VALIDATION set (not test) to avoid consuming the held-out budget.

3. SHAP (SHapley Additive exPlanations) — game-theoretic attribution that
   satisfies efficiency, symmetry, and dummy axioms. TreeExplainer is used
   for RF and XGBoost (exact, not approximate). Applied to a subsample to
   limit memory and compute time.

4. XGBoost gain importance — built-in gain-based importance that measures
   the average improvement in loss from each feature's splits.

Limitations and Caveats (documented in-code)
---------------------------------------------
- MDI is biased toward high-cardinality numerical features.
- Permutation importance underestimates correlated features (PM2.5 and PM10
  are highly correlated; permuting one leaves information in the other).
- SHAP for multiclass produces per-class attributions; we summarise by
  mean |SHAP| across classes for a single global ranking.
- Importance ≠ causation. A feature ranked highly may be a proxy for
  an unmeasured confounder (e.g., Month proxies for temperature inversion
  patterns not directly measured in this dataset).
- This dataset contains NO meteorological fields (temperature, wind speed,
  humidity, pressure). Any temporal feature importance (Month, Season) likely
  captures seasonal pollution patterns rather than direct weather causation.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "figure.dpi":      150,
    "savefig.dpi":     150,
    "font.family":     "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Clean feature names (strip pipeline prefix)
def _clean_name(name: str) -> str:
    for prefix in ("numerical__", "categorical__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


# ─────────────────────────────────────────────────────────────────
# 1. MDI / Gain Importance (from fitted tree models)
# ─────────────────────────────────────────────────────────────────

def get_tree_importance(
    pipeline,
    feature_names: List[str],
    importance_type: str = "mdi",
) -> pd.DataFrame:
    """
    Extract feature importances from a fitted tree-based pipeline.

    Args
    ----
    pipeline       : Fitted sklearn Pipeline with named step 'classifier'.
    feature_names  : Ordered list of feature names from the preprocessor output.
    importance_type: 'mdi' for sklearn MDI, 'gain' for XGBoost gain.

    Returns
    -------
    pd.DataFrame with columns: feature, importance, rank.
    """
    clf = pipeline.named_steps["classifier"]

    if not hasattr(clf, "feature_importances_"):
        raise AttributeError(f"Classifier {type(clf).__name__} has no feature_importances_.")

    importances = clf.feature_importances_
    if len(feature_names) != len(importances):
        raise ValueError("Feature names and tree importance dimensions differ")
    n = len(feature_names)

    df = pd.DataFrame({
        "feature":    [_clean_name(f) for f in feature_names[:n]],
        "importance": importances[:n],
        "type":       importance_type,
    })
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


# ─────────────────────────────────────────────────────────────────
# 2. Permutation Importance
# ─────────────────────────────────────────────────────────────────

def get_permutation_importance(
    pipeline,
    X_val:         pd.DataFrame,
    y_val:         pd.Series,
    feature_names: List[str],
    n_repeats:     int = 10,
    random_state:  int = 42,
    n_jobs:        int = -1,
    is_xgboost:    bool = False,
    preproc_pipeline=None,
) -> pd.DataFrame:
    """
    Compute permutation importance on the validation set.

    The validation set is used (not test) to avoid consuming the held-out
    evaluation budget. Permutation importance measures: for each feature,
    how much does macro F1 drop when that feature's values are randomly
    shuffled across validation rows?

    For XGBoost (early-stopping model), we use the pre-transformed array
    and the raw classifier rather than the full pipeline.

    Args
    ----
    pipeline        : Fitted sklearn Pipeline.
    X_val           : Validation feature DataFrame.
    y_val           : Validation labels.
    feature_names   : Output feature names from the preprocessor.
    n_repeats       : Number of shuffle repetitions per feature.
    is_xgboost      : If True, use preproc_pipeline + raw clf.
    preproc_pipeline: Required if is_xgboost=True.

    Returns
    -------
    pd.DataFrame with columns: feature, importance_mean, importance_std, rank.
    """
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import make_scorer, f1_score

    scorer = make_scorer(f1_score, labels=list(range(6)), average="macro", zero_division=0)

    if is_xgboost and preproc_pipeline is not None:
        clf = pipeline.named_steps["classifier"]
        X_val_t = preproc_pipeline.transform(X_val)
        result = permutation_importance(
            clf, X_val_t, y_val,
            n_repeats=n_repeats,
            scoring=scorer,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    else:
        result = permutation_importance(
            pipeline, X_val, y_val,
            n_repeats=n_repeats,
            scoring=scorer,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    if len(feature_names) != len(result.importances_mean):
        raise ValueError("Feature names and permutation dimensions differ")
    n = len(feature_names)
    df = pd.DataFrame({
        "feature":          [_clean_name(f) for f in feature_names[:n]],
        "importance_mean":  result.importances_mean[:n],
        "importance_std":   result.importances_std[:n],
        "type":             "permutation",
    })
    df = df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


# ─────────────────────────────────────────────────────────────────
# 3. SHAP Explanations
# ─────────────────────────────────────────────────────────────────

def get_shap_values(
    pipeline,
    X_sample: np.ndarray,
    feature_names: List[str],
    is_xgboost: bool = False,
    preproc_pipeline=None,
    X_raw_sample: Optional[pd.DataFrame] = None,
    n_background: int = 200,
    *,
    X_background: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Legacy array API, requiring an explicit training-only SHAP background.

    Both arrays must already be transformed by this fitted pipeline. Missing
    SHAP or unsupported explanations raise an explicit error for caller reporting.
    The canonical raw-DataFrame API is shap_importance_with_background below.
    """
    if X_background is None:
        raise ValueError("An explicit training-only X_background array is required")
    import shap
    background = np.asarray(X_background)
    if len(background) > n_background:
        indices = np.random.RandomState(42).choice(len(background), n_background, replace=False)
        background = background[indices]
    explainer = shap.TreeExplainer(pipeline.named_steps["classifier"], data=background,
                                  feature_perturbation="interventional", model_output="raw")
    values = explainer.shap_values(X_sample, check_additivity=True)
    values = np.stack(values, axis=-1) if isinstance(values, list) else np.asarray(values)
    if values.ndim == 2:
        values = values[..., np.newaxis]
    if values.shape[:2] != np.asarray(X_sample).shape or values.shape[1] != len(feature_names):
        raise ValueError("SHAP output dimensions do not match input features")
    return np.abs(values).mean(axis=(0, 2)), [_clean_name(f) for f in feature_names]


def build_shap_df(
    mean_abs_shap: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """Build a sorted DataFrame from mean |SHAP| values."""
    if len(mean_abs_shap) == 0:
        return pd.DataFrame()
    df = pd.DataFrame({
        "feature":    feature_names,
        "importance": mean_abs_shap,
        "type":       "shap",
    })
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


# ─────────────────────────────────────────────────────────────────
# 4. Plotting Functions
# ─────────────────────────────────────────────────────────────────

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_importance_bar(
    df:         pd.DataFrame,
    title:      str,
    out_dir:    Path,
    filename:   str,
    colour:     str = "#4C72B0",
    top_n:      int = 18,
    x_col:      str = "importance_mean",
    err_col:    Optional[str] = "importance_std",
) -> Path:
    """
    Horizontal bar chart of feature importances (top N features).

    Bars are sorted descending so the most important feature is at the top.
    Error bars are shown if err_col is present and not None.
    """
    _ensure_dir(out_dir)

    # Use correct column name
    if x_col not in df.columns and "importance" in df.columns:
        x_col = "importance"
        err_col = None

    plot_df = df.head(top_n).copy().iloc[::-1]  # reverse for horizontal bar

    xerr = None
    if err_col and err_col in plot_df.columns:
        xerr = plot_df[err_col].values

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.42)))
    bars = ax.barh(
        plot_df["feature"], plot_df[x_col],
        xerr=xerr,
        color=colour, alpha=0.85,
        edgecolor="white", linewidth=0.5,
        error_kw={"elinewidth": 0.8, "capsize": 2, "color": "grey"},
    )
    ax.set_xlabel("Importance Score", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.grid(axis="x", alpha=0.35)

    # Value labels
    for bar, val in zip(bars, plot_df[x_col]):
        ax.text(
            bar.get_width() + bar.get_width() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", fontsize=7, color="#333333",
        )

    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [CHART] {out_path.name}")
    return out_path


def plot_importance_comparison(
    dfs:      Dict[str, pd.DataFrame],
    title:    str,
    out_dir:  Path,
    filename: str,
    top_n:    int = 12,
    x_col:    str = "importance",
) -> Path:
    """
    Side-by-side grouped bar chart comparing importance across multiple methods/models.

    Args
    ----
    dfs    : Dict mapping method label → importance DataFrame (with 'feature' col).
    top_n  : Number of features to show (union of top features across all methods).
    """
    _ensure_dir(out_dir)

    # Get union of top features across all dfs
    all_features: List[str] = []
    for df in dfs.values():
        col = x_col if x_col in df.columns else "importance_mean"
        top = df.nlargest(top_n, col)["feature"].tolist()
        all_features.extend(top)
    # Deduplicate while preserving order
    seen = set()
    features = [f for f in all_features if not (f in seen or seen.add(f))][:top_n]

    methods = list(dfs.keys())
    n_methods = len(methods)
    x = np.arange(len(features))
    bar_width = 0.8 / n_methods

    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:n_methods]

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (method, df) in enumerate(dfs.items()):
        col = x_col if x_col in df.columns else "importance_mean"
        feat_to_imp = dict(zip(df["feature"], df[col]))
        vals = [feat_to_imp.get(f, 0.0) for f in features]
        offset = (i - n_methods / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width * 0.9, label=method,
               color=colours[i], alpha=0.85, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Importance Score", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.legend(title="Method", fontsize=9, title_fontsize=9)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()

    out_path = out_dir / filename
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [CHART] {out_path.name}")
    return out_path


def plot_shap_beeswarm(
    shap_values_by_class: Any,
    X_sample_df:          pd.DataFrame,
    out_dir:              Path,
    filename:             str = "shap_beeswarm.png",
    max_display:          int = 15,
) -> Optional[Path]:
    """
    SHAP beeswarm summary plot using the shap library's built-in visualisation.
    Aggregates across classes by mean |SHAP|.
    """
    try:
        import shap
    except ImportError:
        return None

    _ensure_dir(out_dir)

    try:
        fig, ax = plt.subplots(figsize=(9, max(5, max_display * 0.4)))

        if isinstance(shap_values_by_class, list):
            # Mean |SHAP| across classes
            mean_shap = np.mean(np.abs(np.stack(shap_values_by_class, axis=-1)), axis=-1)
        elif shap_values_by_class.ndim == 3:
            mean_shap = np.mean(np.abs(shap_values_by_class), axis=-1)
        else:
            mean_shap = np.abs(shap_values_by_class)

        shap.summary_plot(
            mean_shap,
            X_sample_df,
            max_display=max_display,
            show=False,
            plot_type="bar",
        )
        out_path = out_dir / filename
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close("all")
        print(f"  [CHART] {out_path.name}")
        return out_path
    except Exception as e:
        print(f"  [SHAP beeswarm] Failed: {e}")
        return None


def plot_cv_score_distribution(
    cv_results_df:  pd.DataFrame,
    model_name:     str,
    out_dir:        Path,
    filename_prefix: str = "",
) -> Path:
    """
    Plot distribution of cross-validation macro F1 scores across all hyperparameter
    configurations explored during tuning.
    """
    _ensure_dir(out_dir)

    score_col = "mean_test_score"
    if score_col not in cv_results_df.columns:
        return None

    scores = cv_results_df[score_col].dropna().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(scores)), scores.values, color="#4C72B0", alpha=0.75, width=0.8)
    ax.axhline(scores.iloc[0], color="#C44E52", linestyle="--", linewidth=1.2,
               label=f"Best CV F1: {scores.iloc[0]:.4f}")
    ax.set_xlabel("Configuration Index (sorted by score)", fontsize=10)
    ax.set_ylabel("CV Macro F1 Score", fontsize=10)
    ax.set_title(f"{model_name} — Cross-Validation Score Distribution",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()

    safe = model_name.lower().replace(" ", "_")
    out_path = out_dir / f"{filename_prefix}cv_scores_{safe}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [CHART] {out_path.name}")
    return out_path


def shap_importance_with_background(pipeline, X_background, X_explain):
    """Interventional TreeSHAP in raw model-output units, with training background.

    Raises on unavailable/unsupported SHAP instead of returning fabricated scores.
    Background and evaluation frames are transformed by the fitted model itself.
    """
    import shap
    preproc = pipeline[:-1]
    background = preproc.transform(X_background)
    explain = preproc.transform(X_explain)
    names = [_clean_name(x) for x in preproc.get_feature_names_out()]
    explainer = shap.TreeExplainer(pipeline.named_steps["classifier"], data=background,
                                  feature_perturbation="interventional", model_output="raw")
    values = explainer.shap_values(explain, check_additivity=True)
    values = np.stack(values, axis=-1) if isinstance(values, list) else np.asarray(values)
    if values.ndim == 2:
        values = values[..., np.newaxis]
    if values.shape[:2] != explain.shape:
        raise ValueError(f"Unexpected SHAP shape {values.shape}; features {explain.shape}")
    frame = pd.DataFrame({"feature": names, "importance": np.abs(values).mean(axis=(0, 2))})
    return frame.sort_values("importance", ascending=False).reset_index(drop=True), values

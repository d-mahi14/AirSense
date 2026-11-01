"""Reproducible feature analysis using training background and validation only."""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
from src.model_io import load_pipeline
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiment import (MODELS, REPORTS, HISTORY, SURVEY, data_for, read, dump,
                            validate_manifest, markdown_table, sha)
from src.explainability import (get_tree_importance, get_permutation_importance,
                               shap_importance_with_background)

PERM_ROWS = 500
PERM_REPEATS = 3
SHAP_BACKGROUND_ROWS = 80
SHAP_EVAL_ROWS = 120


def plot(frame, column, title, path, std=None):
    frame = frame.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(frame.feature, frame[column], xerr=None if std is None else frame[std], color="#218b95")
    ax.set_xlabel("Mean change in macro F1" if column == "importance_mean" else "Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    train_manifest = read(MODELS / "train_manifest.json")
    validate_manifest(train_manifest)
    _, Xtrain, _, _ = data_for("train")
    _, Xval, yval, _ = data_for("validation")
    rng = np.random.RandomState(42)
    perm_idx = rng.choice(len(Xval), min(PERM_ROWS, len(Xval)), replace=False)
    background_idx = rng.choice(len(Xtrain), min(SHAP_BACKGROUND_ROWS, len(Xtrain)), replace=False)
    shap_idx = rng.choice(len(Xval), min(SHAP_EVAL_ROWS, len(Xval)), replace=False)
    figdir = REPORTS / "figures" / "feature_analysis"
    figdir.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "feature_analysis"
    out.mkdir(exist_ok=True)
    for path in list(figdir.glob("*.png")) + list(out.glob("*.csv")) + list(out.glob("*.npz")):
        path.unlink()
    manifest = {"experiment": train_manifest["experiment"],
                "permutation_rows": len(perm_idx), "permutation_repeats": PERM_REPEATS,
                "shap_training_background_rows": len(background_idx), "shap_validation_rows": len(shap_idx),
                "permutation_validation_indices": perm_idx.tolist(),
                "shap_background_training_indices": background_idx.tolist(),
                "shap_validation_indices": shap_idx.tolist(), "models": {}}
    dump(out / "sampling_manifest.json", manifest)
    lines = ["# AirSense — Feature analysis", "", HISTORY, "",
             "## Scope and actual settings", "",
             f"Baseline RF and XGBoost pipelines. Permutation: {len(perm_idx)} validation rows, "
             f"{PERM_REPEATS} repeats, macro F1 across all six classes, seed 42. "
             f"XGBoost SHAP: {len(background_idx)} training background rows, {len(shap_idx)} validation rows. "
             "RF SHAP is not attempted to limit computation; RF MDI and permutation are provided. "
             "Actual row indices, model hashes, failures and importance CSVs are saved in `reports/feature_analysis/`. "
             "Validation was also used for model selection; this analysis is exploratory, not an independent "
             "performance estimate. Test rows are not used.", "",
             "## Predictors", "", ", ".join(Xtrain.columns), "",
             "City is excluded, so no location-importance conclusion is possible. "
             "Weather fields are absent; calendar features do not measure weather effects. "
             "Importance is predictive association, not causation.", ""]
    for key, label, kind in [("random_forest", "Random Forest", "mdi"), ("xgboost", "XGBoost", "gain")]:
        model_result = {}
        manifest["models"][key] = model_result
        lines += [f"## {label}", ""]
        try:
            record = train_manifest["candidates"][key]
            if "error" in record:
                raise RuntimeError(record["error"])
            pipe = load_pipeline(MODELS / f"{key}.joblib")
            model_result["model_sha256"] = sha(MODELS / f"{key}.joblib")
            names = pipe[:-1].get_feature_names_out().tolist()
            tree = get_tree_importance(pipe, names, kind)
            tree.to_csv(out / f"{key}_{kind}.csv", index=False)
            plot(tree, "importance", f"{label}: {kind.upper()} importance", figdir / f"{key}_{kind}.png")
            lines += [f"### {kind.upper()}", "", markdown_table(tree.head(12)), "",
                      f"![{kind}](figures/feature_analysis/{key}_{kind}.png)", ""]
            model_result["top_tree_feature"] = tree.feature.iloc[0]
            perm = get_permutation_importance(pipe, Xval.iloc[perm_idx], yval.iloc[perm_idx],
                                              Xtrain.columns.tolist(), n_repeats=PERM_REPEATS,
                                              random_state=42, n_jobs=2)
            perm.to_csv(out / f"{key}_permutation.csv", index=False)
            plot(perm, "importance_mean", f"{label}: validation permutation", figdir / f"{key}_permutation.png",
                 std="importance_std")
            lines += ["### Permutation", "", markdown_table(perm.head(12)), "",
                      f"![Permutation](figures/feature_analysis/{key}_permutation.png)", "",
                      f"Largest measured drop: **{perm.feature.iloc[0]}**, {perm.importance_mean.iloc[0]:.4f}. "
                      "Error bars are variation across shuffles, not confidence intervals.", ""]
            model_result["top_permutation_feature"] = perm.feature.iloc[0]
            if key == "xgboost":
                try:
                    shap_frame, values = shap_importance_with_background(pipe, Xtrain.iloc[background_idx], Xval.iloc[shap_idx])
                    shap_frame.to_csv(out / f"{key}_shap.csv", index=False)
                    np.savez_compressed(out / "xgboost_shap_values.npz", values=values)
                    plot(shap_frame, "importance", "XGBoost: mean absolute TreeSHAP", figdir / "xgboost_shap.png")
                    lines += ["### SHAP", "", markdown_table(shap_frame.head(12)), "",
                              "![SHAP](figures/feature_analysis/xgboost_shap.png)", "",
                              "Mean absolute interventional TreeSHAP is averaged over validation rows and classes "
                              "in raw margin units, not probability changes. Training background defines the reference distribution.", ""]
                    model_result["shap_status"] = "completed"
                except Exception as exc:
                    model_result["shap_error"] = f"{type(exc).__name__}: {exc}"
                    lines += [f"SHAP unavailable/failed: {model_result['shap_error']}", ""]
        except Exception as exc:
            model_result["error"] = f"{type(exc).__name__}: {exc}"
            lines += [f"Feature analysis failed: {model_result['error']}", ""]
        dump(out / "sampling_manifest.json", manifest)
    corr = Xtrain.select_dtypes(include="number").corr(method="spearman")
    corr.to_csv(out / "training_spearman_correlations.csv")
    pairs = []
    for i, name in enumerate(corr.columns):
        for j in range(i + 1, len(corr.columns)):
            value = corr.iloc[i, j]
            if pd.notna(value):
                pairs.append({"feature_1": name, "feature_2": corr.columns[j], "training_spearman_rho": value})
    pairs = pd.DataFrame(pairs).sort_values("training_spearman_rho", key=lambda s: s.abs(), ascending=False)
    lines += ["## Measured training correlations", "", markdown_table(pairs.head(10)), "",
              "MDI can favor variables with many possible splits. Gain is model-specific. "
              "Correlated features can substitute for each other, lowering single-feature permutation importance; "
              "shuffling can also produce unrealistic pollutant combinations. SHAP depends on the background and "
              "feature-dependence assumptions and does not resolve confounding. Calendar covariates are correlated "
              "by construction. No NOx identity or numerical correlation is assumed without measuring the supplied data. "
              "Contemporaneous pollutants help define AQI, so their importance partly reflects the target construction. "
              "Missingness, sensor coverage, selection of labeled city-days and temporal drift can bias importance. "
              "Only two baseline models and a small validation sample are analyzed; stability across periods/seeds "
              "and geographic transfer remain untested.", "",
              "## Primary-survey context", "", SURVEY, "",
              "The survey's predictor taxonomy motivates documenting available inputs. These importance results "
              "cannot establish what absent weather inputs would contribute. No additional unverified references are cited.", "",
              "## Reproduce", "", "`python -m scripts.analyze_features` after training. "
              "Install the declared SHAP dependency to attempt SHAP; unsupported explanations are recorded as failures."]
    (REPORTS / "feature_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    main()

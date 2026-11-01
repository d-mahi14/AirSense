"""
AirSense — Run All Models Script
==================================
Trains and evaluates all four models on the actual city_day.csv dataset.
Records results to reports/model_results.md.
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Point data loader to the actual data directory
_data_dir = Path(__file__).parent.parent / "data"
os.environ.setdefault("AIRSENSE_DATA_DIR", str(_data_dir))

from src.data_loader import load_city_day
from src.preprocessing import filter_labeled_data
from src.data_split import stratified_data_split

print("=" * 65)
print("  AirSense — Full Model Training & Evaluation Run")
print("=" * 65)

# ── 1. Load and split data ─────────────────────────────────────
print("\n[1/5] Loading dataset...")
df = load_city_day()
df, audit = filter_labeled_data(df)
print(f"  Retained rows: {audit['retained_rows']:,}  |  Dropped (no label): {audit['dropped_missing_target']:,}")

print("\n[2/5] Creating stratified splits (70/15/15, seed=42)...")
train_df, val_df, test_df = stratified_data_split(df, random_state=42)
print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")
print(f"  Class distribution in train:")
for cat, cnt in train_df["AQI_Bucket"].value_counts().sort_index().items():
    pct = cnt / len(train_df) * 100
    print(f"    {cat}: {cnt:,} ({pct:.1f}%)")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

all_results = {}

# ── 2. Decision Tree ───────────────────────────────────────────
print("\n[3/5] Decision Tree")
print("-" * 40)
from src.models.decision_tree_model import run_decision_tree
dt_results = run_decision_tree(train_df, val_df, test_df, model_dir=MODEL_DIR)
all_results["decision_tree"] = {
    "val_accuracy": dt_results["val_metrics"]["accuracy"],
    "test_accuracy": dt_results["test_metrics"]["accuracy"],
    "val_macro_f1": dt_results["val_metrics"]["macro_f1"],
    "test_macro_f1": dt_results["test_metrics"]["macro_f1"],
    "train_time_sec": dt_results["train_time_sec"],
}

# ── 3. Random Forest ───────────────────────────────────────────
print("\n[3b/5] Random Forest")
print("-" * 40)
from src.models.random_forest_model import run_random_forest
rf_results = run_random_forest(train_df, val_df, test_df, model_dir=MODEL_DIR)
all_results["random_forest"] = {
    "val_accuracy": rf_results["val_metrics"]["accuracy"],
    "test_accuracy": rf_results["test_metrics"]["accuracy"],
    "val_macro_f1": rf_results["val_metrics"]["macro_f1"],
    "test_macro_f1": rf_results["test_metrics"]["macro_f1"],
    "train_time_sec": rf_results["train_time_sec"],
    "oob_score": rf_results["oob_score"],
}

# ── 4. SVM ─────────────────────────────────────────────────────
print("\n[4/5] SVM (RBF kernel, full training set)")
print("-" * 40)
from src.models.svm_model import run_svm
svm_results = run_svm(
    train_df, val_df, test_df,
    model_dir=MODEL_DIR,
    max_train_samples=None  # Full dataset — may take several minutes
)
all_results["svm"] = {
    "val_accuracy": svm_results["val_metrics"]["accuracy"],
    "test_accuracy": svm_results["test_metrics"]["accuracy"],
    "val_macro_f1": svm_results["val_metrics"]["macro_f1"],
    "test_macro_f1": svm_results["test_metrics"]["macro_f1"],
    "train_time_sec": svm_results["train_time_sec"],
    "subsampled": svm_results["subsampled"],
}

# ── 5. XGBoost ─────────────────────────────────────────────────
print("\n[5/5] XGBoost (with early stopping on validation set)")
print("-" * 40)
from src.models.xgboost_model import run_xgboost
xgb_results = run_xgboost(
    train_df, val_df, test_df,
    model_dir=MODEL_DIR,
    use_early_stopping=True,
    early_stopping_rounds=30
)
all_results["xgboost"] = {
    "val_accuracy": xgb_results["val_metrics"]["accuracy"],
    "test_accuracy": xgb_results["test_metrics"]["accuracy"],
    "val_macro_f1": xgb_results["val_metrics"]["macro_f1"],
    "test_macro_f1": xgb_results["test_metrics"]["macro_f1"],
    "train_time_sec": xgb_results["train_time_sec"],
    "best_iteration": xgb_results["best_iteration"],
}

# ── Summary ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  FINAL RESULTS SUMMARY")
print("=" * 65)
print(f"\n{'Model':<18} {'Val Acc':>10} {'Test Acc':>10} {'Val F1':>10} {'Test F1':>10} {'Train(s)':>10}")
print("-" * 65)
for name, r in all_results.items():
    print(f"{name:<18} {r['val_accuracy']:>10.4f} {r['test_accuracy']:>10.4f} "
          f"{r['val_macro_f1']:>10.4f} {r['test_macro_f1']:>10.4f} {r['train_time_sec']:>10.2f}")

# Save results
results_path = Path("models") / "all_results.json"
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to: {results_path}")

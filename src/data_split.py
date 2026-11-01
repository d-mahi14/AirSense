"""
AirSense Data Splitting Module
==============================
Provides disciplined, reproducible splitting strategies for the AirSense
air quality classification benchmark:

1. Stratified Split (Primary Benchmark):
   - Preserves exact CPCB class proportions (70% Train, 15% Val, 15% Test).
   - Ideal for fair algorithmic comparisons across severe class imbalances.
2. Temporal / Chronological Split:
   - Partitions data chronologically across time (e.g. 2015 to mid-2019 Train,
     late 2019 Val, 2020 Test).
   - Evaluates temporal generalization and secular shifts.
3. Spatial / Grouped Split:
   - Groups by City, ensuring no city appears in both training and test sets.
   - Evaluates out-of-domain geographic transferability.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np

from src.data_loader import CPCB_AQI_CATEGORIES


def stratified_data_split(
    df: pd.DataFrame,
    target_col: str = "AQI_Bucket",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Primary Benchmark Split: Stratified 70% Train, 15% Val, 15% Test.

    Guarantees exact class proportions across all six CPCB categories.
    Reproducible via fixed random seed (default 42).
    """
    rng = np.random.RandomState(random_state)
    train_idx_all = []
    val_idx_all = []
    test_idx_all = []

    for _, group in df.groupby(target_col):
        indices = group.index.values.copy()
        rng.shuffle(indices)

        n = len(indices)
        n_test = int(round(n * test_size))
        n_val = int(round(n * val_size))

        test_idx = indices[:n_test]
        val_idx = indices[n_test : n_test + n_val]
        train_idx = indices[n_test + n_val :]

        test_idx_all.extend(test_idx)
        val_idx_all.extend(val_idx)
        train_idx_all.extend(train_idx)

    train_df = df.loc[train_idx_all].copy().sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    val_df = df.loc[val_idx_all].copy().sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test_df = df.loc[test_idx_all].copy().sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return train_df, val_df, test_df


def temporal_data_split(
    df: pd.DataFrame,
    date_col: str = "Date",
    train_end: str = "2019-06-30",
    val_end: str = "2019-12-31"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological / Temporal Split:
    - Train: All records with Date <= train_end (e.g., 2015-01-01 to 2019-06-30)
    - Val:   train_end < Date <= val_end (e.g., 2019-07-01 to 2019-12-31)
    - Test:  Date > val_end (e.g., 2020-01-01 to 2020-07-01)

    Respects the arrow of time without future look-ahead bias.
    """
    df_temp = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_temp[date_col]):
        df_temp[date_col] = pd.to_datetime(df_temp[date_col])

    dt_train_end = pd.to_datetime(train_end)
    dt_val_end = pd.to_datetime(val_end)

    train_df = df_temp[df_temp[date_col] <= dt_train_end].copy().reset_index(drop=True)
    val_df = df_temp[(df_temp[date_col] > dt_train_end) & (df_temp[date_col] <= dt_val_end)].copy().reset_index(drop=True)
    test_df = df_temp[df_temp[date_col] > dt_val_end].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def spatial_data_split(
    df: pd.DataFrame,
    group_col: str = "City",
    test_cities: Optional[List[str]] = None,
    val_cities: Optional[List[str]] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Spatial / Grouped Split (Leave-Cities-Out):
    Partitions data so that entire cities are reserved exclusively for
    validation and testing. Evaluates cross-city geographic transferability.
    """
    all_cities = sorted(df[group_col].dropna().unique().tolist())
    rng = np.random.RandomState(random_state)

    if test_cities is None or val_cities is None:
        shuffled = all_cities.copy()
        rng.shuffle(shuffled)
        n_cities = len(shuffled)
        n_test = max(2, int(round(n_cities * 0.15)))
        n_val = max(2, int(round(n_cities * 0.15)))

        test_cities = shuffled[:n_test]
        val_cities = shuffled[n_test : n_test + n_val]
        train_cities = shuffled[n_test + n_val :]
    else:
        train_cities = [c for c in all_cities if c not in test_cities and c not in val_cities]

    train_df = df[df[group_col].isin(train_cities)].copy().reset_index(drop=True)
    val_df = df[df[group_col].isin(val_cities)].copy().reset_index(drop=True)
    test_df = df[df[group_col].isin(test_cities)].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def audit_data_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "AQI_Bucket",
    date_col: Optional[str] = "Date"
) -> Dict[str, Any]:
    """
    Perform a complete audit on a train/val/test split partition.
    Verifies sample counts, class balance, and checks for overlap.
    """
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    n_total = n_train + n_val + n_test

    # Category proportions
    def get_class_dist(df_part):
        if target_col not in df_part.columns:
            return {}
        counts = df_part[target_col].value_counts().to_dict()
        pcts = {cat: round((counts.get(cat, 0) / len(df_part)) * 100, 2) for cat in CPCB_AQI_CATEGORIES}
        return pcts

    train_pcts = get_class_dist(train_df)
    val_pcts = get_class_dist(val_df)
    test_pcts = get_class_dist(test_df)

    # Date ranges if available
    date_info = {}
    if date_col and date_col in train_df.columns:
        date_info = {
            "train_range": (str(train_df[date_col].min()), str(train_df[date_col].max())),
            "val_range": (str(val_df[date_col].min()), str(val_df[date_col].max())),
            "test_range": (str(test_df[date_col].min()), str(test_df[date_col].max()))
        }

    # Reset row indices are not record identities. Check real city/date keys.
    overlap_check = {"checked": False, "reason": "City and Date are required for identity checking"}
    if all("City" in part.columns and "Date" in part.columns for part in [train_df, val_df, test_df]):
        def record_keys(part):
            dates = pd.to_datetime(part["Date"], errors="raise").dt.normalize()
            if part["City"].isna().any() or dates.isna().any():
                raise ValueError("Cannot audit split identity with missing City or Date")
            return set(zip(part["City"].astype(str), dates))
        train_keys, val_keys, test_keys = map(record_keys, [train_df, val_df, test_df])
        counts = {"train_validation": len(train_keys & val_keys),
                  "train_test": len(train_keys & test_keys),
                  "validation_test": len(val_keys & test_keys)}
        overlap_check = {"checked": True, "identity_columns": ["City", "Date"],
                         "overlap_counts": counts, "disjoint": not any(counts.values())}

    return {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_total": n_total,
        "overlap_check": overlap_check,
        "train_pct": round((n_train / n_total) * 100, 2) if n_total > 0 else 0.0,
        "val_pct": round((n_val / n_total) * 100, 2) if n_total > 0 else 0.0,
        "test_pct": round((n_test / n_total) * 100, 2) if n_total > 0 else 0.0,
        "class_distributions": {
            "train": train_pcts,
            "val": val_pcts,
            "test": test_pcts
        },
        "date_info": date_info
    }


def save_split_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    strategy_name: str = "stratified",
    output_dir: Union[str, Path] = "data/processed"
) -> Dict[str, Path]:
    """
    Persist splits for a specific strategy under data/processed/<strategy_name>/.
    """
    out_path = Path(output_dir) / strategy_name
    out_path.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": out_path / "train.csv",
        "val": out_path / "val.csv",
        "test": out_path / "test.csv"
    }

    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"], index=False)
    test_df.to_csv(paths["test"], index=False)

    return paths


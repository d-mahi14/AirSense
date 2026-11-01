"""Bounded macro-F1 searches with date-blocked, expanding-window CV.

Every preprocessor is cloned and fitted inside each fold. The search functions
accept training data only; validation and test data are not search inputs.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV

RANDOM_STATE = 42
MACRO_F1_SCORER = make_scorer(f1_score, labels=list(range(6)), average="macro", zero_division=0)


def date_blocked_cv(dates, n_splits=3, gap_days=7):
    """Expanding windows on unique dates; same-day cities never span a fold.

    First 40% of dates warm up the model; the remaining dates form disjoint
    validation windows. Training ends at least gap_days before each window.
    City overlap is intentional: this evaluates later dates in observed cities,
    not geographic transfer. A separate leave-city-out study is still needed.
    """
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    unique = np.sort(dates.unique())
    start = max(1, int(len(unique) * 0.4))
    windows = np.array_split(unique[start:], n_splits)
    folds = []
    for window in windows:
        if not len(window):
            raise ValueError("Too few distinct dates for temporal CV")
        cutoff = pd.Timestamp(window[0]) - pd.Timedelta(days=gap_days)
        tr = np.flatnonzero((dates < cutoff).to_numpy())
        va = np.flatnonzero(dates.isin(window).to_numpy())
        if not len(tr) or not len(va):
            raise ValueError("Empty temporal fold")
        folds.append((tr, va))
    return folds


def get_search_spaces():
    """Finite, serializable spaces; budget is six candidates × three folds."""
    return {
        "decision_tree": {
            "classifier__max_depth": [5, 8, 10, 15],
            "classifier__min_samples_leaf": [5, 10, 20],
            "classifier__ccp_alpha": [0.0, 0.0005],
        },
        "random_forest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [10, 15, None],
            "classifier__min_samples_leaf": [2, 5, 10],
            "classifier__max_features": ["sqrt", 0.5],
        },
        "svm": {
            "classifier__C": [0.1, 1.0, 5.0, 10.0, 30.0],
            "classifier__gamma": ["scale", "auto"],
        },
        "xgboost": {
            "classifier__n_estimators": [150, 300],
            "classifier__max_depth": [4, 6, 8],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__reg_lambda": [1.0, 2.0],
        },
    }


def run_search(pipeline, space, X, y, dates, n_iter=6, n_jobs=2):
    folds = date_blocked_cv(dates)
    for tr, _ in folds:
        if set(np.asarray(y)[tr]) != set(range(6)):
            raise ValueError("A temporal training fold lacks one or more classes")
    search = RandomizedSearchCV(
        pipeline, space, n_iter=n_iter, scoring=MACRO_F1_SCORER,
        cv=folds, random_state=RANDOM_STATE, n_jobs=n_jobs,
        refit=False, error_score="raise", return_train_score=False,
    )
    start = time.perf_counter()
    search.fit(X, y)
    return search.best_params_, pd.DataFrame(search.cv_results_), time.perf_counter() - start
